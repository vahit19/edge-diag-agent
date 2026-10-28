"""Offline diagnostic agent: retrieve manual entries, answer with a local quantized LLM, check the answer.

    python -m diag.agent "G120C shows F7900 when running in automatic mode" [--model data/models/....gguf]

The answer is generated only from the retrieved List Manual entries and every fault code or
parameter number in it is checked against what the model was shown: the question and the
excerpts (the grounding gate). Numbers it was not shown are reported as unsupported if they
exist elsewhere in the manuals, and as invented if they do not exist in the manuals at all.
"""

import argparse
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from diag.retrieve import CODE, PARAM, Retriever

DEFAULT_MODEL = "data/models/qwen2.5-3b-instruct-q4_k_m.gguf"
SYSTEM = (
    "You are a maintenance assistant for Siemens SINAMICS G120 drives, used offline on the plant floor. "
    "Answer ONLY from the manual excerpts provided. Each excerpt is tagged [S1], [S2], ... "
    "Give: (1) what the fault or symptom means, (2) the most likely causes for this situation, "
    "(3) concrete checks and parameter settings, (4) how to acknowledge, if relevant. "
    "Cite the excerpt tag after each statement, e.g. [S2]. Only mention fault codes and parameter "
    "numbers that appear in the excerpts. If the excerpts do not cover the question, say so plainly "
    "and say which document would (for example the operating instructions). Be short and practical."
)


@dataclass
class Answer:
    question: str
    text: str
    sources: list
    mentioned: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)   # not shown to the model, but in the manuals
    invented: list = field(default_factory=list)      # not shown to the model and not in the manuals
    facts: list = field(default_factory=list)         # matched fault/alarm fields, verbatim from the manual
    ack_conflict: list = field(default_factory=list)  # codes whose stated acknowledgment contradicts the manual
    exact: bool = False                               # the question named a code or parameter found in the manuals
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def grounded(self) -> bool:
        return not self.unsupported and not self.invented

    @property
    def passed(self) -> bool:
        return self.grounded and not self.ack_conflict


def ids_in(text: str) -> list:
    found = [f"{a.upper()}{int(b):05d}" for a, b in CODE.findall(text)]
    found += [f"{a.lower()}{int(b):04d}" for a, b in PARAM.findall(text)]
    return list(dict.fromkeys(found))


def excerpt(d, other_chars: int = 1400, max_chars: int = 1400) -> str:
    """The exactly matched entry is cut at max_chars, every other excerpt at other_chars."""
    limit = max_chars if "exact" in d.why else min(other_chars, max_chars)
    return d.text if len(d.text) <= limit else d.text[:limit] + " ..."


def context_block(docs: list, other_chars: int = 1400) -> str:
    return "\n\n".join(f"[S{i}] ({d.device}, {d.cite()})\n{excerpt(d, other_chars)}" for i, d in enumerate(docs, 1))


def ground(text: str, question: str, docs: list, known: set, other_chars: int = 1400) -> tuple:
    """Split the identifiers in an answer into (mentioned, unsupported, invented).

    An identifier is supported if the model was shown it, in the question or in an excerpt
    (the manuals also name parameters of other SINAMICS families, which are not in this list).
    """
    mentioned = ids_in(text)
    shown = set(ids_in(question)) | {i for d in docs for i in ids_in(excerpt(d, other_chars))} | {d.id for d in docs}
    unsupported = [m for m in mentioned if m not in shown and m in known]
    invented = [m for m in mentioned if m not in shown and m not in known]
    return mentioned, unsupported, invented


ACK_MODES = {"immediately": ("immediately",), "power on": ("power on", "power-on", "power cycle", "switch off and on"),
             "none": ("none", "not required", "no acknowledg", "not necessary", "withdrawn", "automatically")}


def stated_ack(text: str) -> set:
    """Acknowledgment modes the answer states near the word 'acknowledg...'."""
    text = re.sub(r"without (a )?power[ -]on", "", text.lower())  # manual wording for alarms, not an instruction
    windows = " ".join(text[max(0, m.start() - 80):m.start() + 150] for m in re.finditer("acknowledg", text))
    return {mode for mode, words in ACK_MODES.items() if any(w in windows for w in words)}


def facts(docs: list) -> list:
    """Reaction and acknowledgment of each exactly matched fault/alarm, verbatim from the manual."""
    out = []
    for d in docs:
        if d.kind == "message" and "exact" in d.why:
            field_of = dict(re.findall(r"^(Type|Reaction|Acknowledge): (.*)$", d.text, re.M))
            out.append({"code": d.id, "device": d.device, "title": d.title, "type": field_of.get("Type", ""),
                        "reaction": field_of.get("Reaction", ""), "acknowledge": field_of.get("Acknowledge", ""),
                        "cite": d.cite()})
    return out


def ack_conflicts(text: str, fact_list: list) -> list:
    """Codes whose acknowledgment, as stated in the answer, contradicts the manual. Checked only
    when the question matched a single fault/alarm, so the stated mode can belong to nothing else."""
    if len(fact_list) != 1:
        return []
    f = fact_list[0]
    stated = stated_ack(text)
    manual = {mode for mode in ACK_MODES if mode in f["acknowledge"].lower()}
    return [f["code"]] if stated and not stated & manual else []


class Agent:
    def __init__(self, kb: Path = Path("data/kb"), index: Path = Path("data/index"), model: str = DEFAULT_MODEL,
                 n_threads: int = 4, n_ctx: int = 6144, retriever: Retriever = None, k: int = 5,
                 other_chars: int = 1400):
        from llama_cpp import Llama
        self.retriever = retriever or Retriever(kb, index)
        self.llm = Llama(model_path=model, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
        self.model = Path(model).name
        self.k, self.other_chars = k, other_chars  # k=3, 400 chars: a third of the prompt, for 2-core devices

    def generate(self, messages: list, max_tokens: int = 450) -> tuple:
        t = time.perf_counter()
        out = self.llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=max_tokens)
        return out["choices"][0]["message"]["content"].strip(), out["usage"], (time.perf_counter() - t) * 1000

    def ask(self, question: str, device: str = None, closed_book: bool = False) -> Answer:
        t = time.perf_counter()
        docs = [] if closed_book else self.retriever.search(question, k=self.k, device=device)
        retrieve_ms = (time.perf_counter() - t) * 1000
        if closed_book:
            messages = [{"role": "system", "content": SYSTEM.split(" Answer ONLY")[0] +
                         " Give the meaning, likely causes, checks and parameter settings. Be short and practical."},
                        {"role": "user", "content": question}]
        else:
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Manual excerpts:\n\n{context_block(docs, self.other_chars)}\n\nQuestion: {question}"}]
        text, usage, gen_ms = self.generate(messages)
        mentioned, unsupported, invented = ground(text, question, docs, self.retriever.known, self.other_chars)
        fact_list = facts(docs)
        return Answer(
            question=question, text=text,
            sources=[{"tag": f"S{i}", "id": d.id, "device": d.device, "cite": d.cite(), "why": d.why}
                     for i, d in enumerate(docs, 1)],
            mentioned=mentioned,
            unsupported=unsupported, invented=invented,
            facts=fact_list, ack_conflict=ack_conflicts(text, fact_list), exact=any("exact" in d.why for d in docs),
            retrieve_ms=retrieve_ms, generate_ms=gen_ms,
            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=os.environ.get("DIAG_MODEL", DEFAULT_MODEL))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("DIAG_THREADS", 4)))
    ap.add_argument("--closed-book", action="store_true", help="no retrieval, for comparison")
    a = ap.parse_args()
    agent = Agent(model=a.model, n_threads=a.threads)
    ans = agent.ask(a.question, device=a.device, closed_book=a.closed_book)
    print(ans.text, "\n")
    for s in ans.sources:
        print(f"  {s['tag']}: {s['id']} ({s['cite']}) {', '.join(s['why'])}")
    for f in ans.facts:
        print(f"  {f['code']} from the manual: reaction {f['reaction']}, acknowledge {f['acknowledge']} ({f['cite']})")
    print(f"\nmentioned={ans.mentioned} unsupported={ans.unsupported} invented={ans.invented} "
          f"ack_conflict={ans.ack_conflict}")
    print(f"retrieve {ans.retrieve_ms:.0f} ms, generate {ans.generate_ms:.0f} ms, "
          f"{ans.prompt_tokens} prompt + {ans.completion_tokens} completion tokens")


if __name__ == "__main__":
    main()
