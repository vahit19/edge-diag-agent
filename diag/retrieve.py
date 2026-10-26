"""Hybrid retrieval over the parsed manuals: exact identifiers first, then BM25 and dense vectors.

Technicians type fault numbers in many shapes ("F7900", "O.F 7807", "the 1910 error",
"p2040", "P1121"), so identifiers are normalised and looked up exactly before any fuzzy
search. Everything else is ranked by reciprocal-rank fusion of BM25 and a small embedding
model that runs on CPU (BAAI/bge-small-en-v1.5 through ONNX Runtime).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CODE = re.compile(r"\b([FAN])\s?0?(\d{4,5})\b", re.I)
PARAM = re.compile(r"\b([pr])\s?(\d{3,5})\b", re.I)
BARE = re.compile(r"\b(?:fault|error|alarm|o\.?f\.?)\s*(?:code\s*)?(\d{4,5})\b|\b(\d{4,5})\s+(?:fault|error|alarm)\b", re.I)
DEVICES = {"g120c": "SINAMICS G120C", "cu240": "SINAMICS G120 CU240B-2/CU240E-2"}
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def tokens(text: str) -> list:
    return re.findall(r"[a-z]+|\d+", text.lower())


@dataclass
class Doc:
    id: str            # e.g. "F07900" or "p1900"
    device: str
    source: str
    page: int
    kind: str          # "message" | "parameter"
    title: str
    text: str
    score: float = 0.0
    why: list = field(default_factory=list)

    def cite(self) -> str:
        return f"{self.source} p.{self.page}"


def load_docs(kb: Path) -> list:
    docs = []
    for line in open(kb / "messages.jsonl", encoding="utf-8"):
        r = json.loads(line)
        body = (f"{r['code']} {r['title']}\nType: {r['type']}\nMessage class: {r['message_class']}\n"
                f"Reaction: {r['reaction']}\nAcknowledge: {r['acknowledge']}\nCause:\n{r['cause']}\nRemedy:\n{r['remedy']}")
        docs.append(Doc(r["code"], r["device"], r["source"], r["page"], "message", r["title"], body))
    for line in open(kb / "parameters.jsonl", encoding="utf-8"):
        r = json.loads(line)
        body = f"{r['number']} {r['name']}\n{r['text']}"
        docs.append(Doc(r["number"], r["device"], r["source"], r["page"], "parameter",
                        f"{r['name']} / {r['short']}" if r["short"] else r["name"], body))
    return docs


def identifiers(query: str, known: set) -> list:
    """Normalised fault/alarm codes and parameter numbers mentioned in the query, in order."""
    found = []
    for letter, num in CODE.findall(query):
        found.append(f"{letter.upper()}{int(num):05d}")
    for letter, num in PARAM.findall(query):
        found.append(f"{letter.lower()}{int(num):04d}")
    for a, b in BARE.findall(query):
        n = int(a or b)
        found += [c for c in (f"F{n:05d}", f"A{n:05d}") if c in known]
    return [i for i in dict.fromkeys(found) if i in known]


def device_of(query: str):
    q = query.lower()
    if "g120c" in q:
        return DEVICES["g120c"]
    if "cu240" in q:
        return DEVICES["cu240"]
    return None


class Retriever:
    def __init__(self, kb: Path, index_dir: Path, use_dense: bool = True, use_bm25: bool = True,
                 use_exact: bool = True):
        self.docs = load_docs(kb)
        self.known = {d.id for d in self.docs}
        self.use_dense, self.use_bm25, self.use_exact = use_dense, use_bm25, use_exact
        self.bm25 = BM25Okapi([tokens(d.text) for d in self.docs])
        self.embedder, self.vectors = None, None
        if use_dense:
            from fastembed import TextEmbedding
            self.embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(index_dir / "fastembed"))
            cache = index_dir / "vectors.npy"
            if cache.exists() and np.load(cache).shape[0] == len(self.docs):
                self.vectors = np.load(cache)
            else:
                index_dir.mkdir(parents=True, exist_ok=True)
                self.vectors = np.array(list(self.embedder.embed([d.text[:2000] for d in self.docs])), dtype=np.float32)
                np.save(cache, self.vectors)

    def search(self, query: str, k: int = 5, device=None, expand: int = 3) -> list:
        """Top-k documents, followed by up to `expand` parameters that each exactly matched
        fault/alarm entry points to in its Remedy (then Cause) text, as a technician would."""
        hits = self._rank(query, k, device)
        if not expand:
            return hits
        by_key = {(d.id, d.device): d for d in self.docs}
        have = {(h.id, h.device) for h in hits}
        linked = []
        for h in hits:
            if h.kind != "message" or "exact" not in h.why:
                continue
            remedy = h.text.split("\nRemedy:\n", 1)[-1]
            cause = h.text.split("\nCause:\n", 1)[-1].split("\nRemedy:\n", 1)[0]
            refs = [r for r in dict.fromkeys(re.findall(r"\b[pr]\d{4}\b", remedy + "\n" + cause))]
            for ref in refs[:expand]:
                d = by_key.get((ref, h.device))
                if d and (ref, h.device) not in have:
                    have.add((ref, h.device))
                    linked.append(Doc(**{**d.__dict__, "score": 0.0, "why": [f"linked from {h.id}"]}))
        return hits + linked

    def _rank(self, query: str, k: int, device) -> list:
        device = device or device_of(query)
        allowed = np.array([device is None or d.device == device for d in self.docs])
        fused = np.zeros(len(self.docs))
        rankings = []
        if self.use_bm25:
            rankings.append(("bm25", self.bm25.get_scores(tokens(query))))
        if self.use_dense:
            q = np.array(list(self.embedder.query_embed(query))[0], dtype=np.float32)
            rankings.append(("dense", self.vectors @ q))
        for _, scores in rankings:
            order = np.argsort(-np.where(allowed, scores, -np.inf))
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(len(order))
            fused += 1.0 / (60 + ranks)
        ids = identifiers(query, self.known) if self.use_exact else []
        for pos, ident in enumerate(ids):  # exact mentions outrank everything, in the order they appear
            for i, d in enumerate(self.docs):
                if d.id == ident and allowed[i]:
                    fused[i] = 10.0 - pos * 0.01
        fused[~allowed] = -1
        out = []
        seen = set()
        for i in np.argsort(-fused):
            d = self.docs[i]
            key = (d.id, d.device)
            if key in seen:
                continue
            seen.add(key)
            hit = Doc(**{**d.__dict__, "score": float(fused[i]), "why": ["exact"] if fused[i] >= 9 else ["fused"]})
            out.append(hit)
            if len(out) == k:
                break
        return out
