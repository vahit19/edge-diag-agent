"""Evaluate retrieval, answer grounding and resource use of the offline diagnostic agent.

    python -m eval.run_eval retrieval                       # seconds, no LLM
    python -m eval.run_eval answers --model <gguf> [--n 40] # LLM answers, writes results/answers_<model>.jsonl
    python -m eval.run_eval rescore                         # re-check saved answers after a change to the checks

Question sets
  forum   real technician questions from public PLCtalk threads, gold codes/parameters taken
          from the thread's resolution and checked against the manual (data/eval/forum_questions.jsonl)
  manual  every 1 in k fault/alarm entries of both List Manuals (seeded sample), asked as
          "The drive shows <code>": the manual entry itself is the ground truth, so the answer's
          stated acknowledgment and meaning can be checked word for word.

Acknowledgment is scored on faults only (alarms clear themselves, the manual says NONE), as
correct, wrong (a different mode is stated) or not stated. For alarms we only count answers
that wrongly tell the technician to acknowledge.
"""

import argparse
import json
import random
import re
import statistics
import time
from math import sqrt
from pathlib import Path

KB, INDEX, OUT = Path("data/kb"), Path("data/index"), Path("results")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f})" if n else "0/0"


def forum_set() -> list:
    return [json.loads(l) for l in open("data/eval/forum_questions.jsonl", encoding="utf-8")]


def manual_set(n: int, seed: int = 0) -> list:
    msgs = [json.loads(l) for l in open(KB / "messages.jsonl", encoding="utf-8")]
    msgs = [m for m in msgs if m["type"] in ("fault", "alarm")]
    rng = random.Random(seed)
    picked = rng.sample(msgs, n)
    return [{"id": f"manual-{m['code']}-{m['source'][:5]}", "device": m["device"], "gold": [m["code"]], "entry": m,
             "question": f"The drive ({m['device']}) shows {m['code']}. What does it mean and what should I do?"}
            for m in picked]


def retrieval(args) -> None:
    from diag.retrieve import Retriever
    configs = {
        "full (exact + BM25 + dense + linked params)": dict(),
        "no linked parameters": dict(expand=0),
        "no exact identifier lookup": dict(use_exact=False),
        "BM25 only": dict(use_dense=False, use_exact=False, expand=0),
        "dense only": dict(use_bm25=False, use_exact=False, expand=0),
    }
    rows, qs = [], [q for q in forum_set() if q["gold"]]
    base = Retriever(KB, INDEX)
    for name, cfg in configs.items():
        expand = cfg.pop("expand", 3)
        r = base
        if cfg:
            r = Retriever.__new__(Retriever)
            r.__dict__.update(base.__dict__)
            r.use_exact = cfg.get("use_exact", True)
            r.use_dense = cfg.get("use_dense", True)
            r.use_bm25 = cfg.get("use_bm25", True)
        hit = tot = full = 0
        for q in qs:
            ids = [h.id for h in r.search(q["question"], k=5, device=q["device"], expand=expand)]
            got = sum(g in ids for g in q["gold"])
            hit, tot, full = hit + got, tot + len(q["gold"]), full + (got == len(q["gold"]))
        rows.append({"config": name, "gold_items_found": hit, "gold_items": tot,
                     "questions_fully_covered": full, "questions": len(qs)})
        print(f"| {name} | {fmt(hit, tot)} | {fmt(full, len(qs))} |")
    OUT.mkdir(exist_ok=True)
    json.dump(rows, open(OUT / "retrieval.json", "w"), indent=1)


def ack_outcome(entry: dict, text: str):
    """Score the acknowledgment the answer states against the manual entry."""
    from diag.agent import stated_ack
    stated = stated_ack(text)
    if entry["type"] == "alarm":
        return "wrong" if stated & {"immediately", "power on"} else None
    gold = {mode for mode in ("immediately", "power on") if mode in entry["acknowledge"].lower()}
    if stated & gold:
        return "correct"
    return "wrong" if stated else "not stated"


def check(q: dict, text: str, mentioned: list, unsupported: list, invented: list, ack_conflict: list) -> dict:
    low = text.lower()
    row = {"id": q["id"], "grounded": not unsupported and not invented, "unsupported": unsupported,
           "invented": invented, "gold_mentioned": [g for g in q["gold"] if g in mentioned]}
    if "entry" in q:
        e = q["entry"]
        row["ack"] = ack_outcome(e, low)
        row["ack_flagged"] = bool(ack_conflict)
        words = [w for w in re.findall(r"[a-z]{4,}", e["title"].lower()) if w not in ("drive", "unit", "power")]
        row["meaning_overlap"] = sum(w in low for w in words) / len(words) if words else None
    if not q["gold"]:
        row["abstained"] = bool(re.search(r"not (cover|contain|includ|provid|mention)|operating instructions|no information", low))
    return row


def quality(rows: list) -> dict:
    summary = {}
    for s in ("forum", "manual"):
        rs = [r for r in rows if r["set"] == s]
        summary[f"{s}_grounded"] = fmt(sum(r["grounded"] for r in rs), len(rs))
        summary[f"{s}_with_invented_ids"] = fmt(sum(bool(r["invented"]) for r in rs), len(rs))
        g = [r for r in rs if s == "manual" or forum_gold(r["id"])]
        summary[f"{s}_gold_items_mentioned"] = fmt(sum(len(r["gold_mentioned"]) for r in g),
                                                   sum(len(forum_gold(r["id"])) if s == "forum" else 1 for r in g))
    ms = [r for r in rows if r["set"] == "manual"]
    faults = [r["ack"] for r in ms if r["id"].startswith("manual-F")]
    for outcome in ("correct", "wrong", "not stated"):
        summary[f"manual_fault_ack_{outcome.replace(' ', '_')}"] = fmt(faults.count(outcome), len(faults))
    alarms = [r["ack"] for r in ms if r["id"].startswith("manual-A")]
    summary["manual_alarm_told_to_acknowledge"] = fmt(alarms.count("wrong"), len(alarms))
    wrong = [r for r in ms if r["ack"] == "wrong"]
    right = [r for r in ms if r["ack"] != "wrong"]
    summary["ack_gate_flags_wrong"] = fmt(sum(r["ack_flagged"] for r in wrong), len(wrong))
    summary["ack_gate_flags_not_wrong"] = fmt(sum(r["ack_flagged"] for r in right), len(right))
    # none of the 40 sampled entries is an encoder fault; the closed-book model guesses one anyway
    summary["manual_encoder_in_first_300_chars"] = fmt(sum("encoder" in r["answer"][:300].lower() for r in ms), len(ms))
    summary["manual_meaning_overlap_mean"] = round(statistics.mean(r["meaning_overlap"] for r in ms if r.get("meaning_overlap") is not None), 2)
    summary["forum_out_of_scope_abstained"] = [r.get("abstained") for r in rows if "abstained" in r]
    return summary


def variant(args) -> str:
    """File-name suffix for settings that differ from the defaults (4 threads, k=5, 1400-char excerpts)."""
    return (("_closedbook" if args.closed_book else "") + (f"_t{args.threads}" if args.threads != 4 else "")
            + (f"_k{args.k}" if args.k != 5 else "") + (f"_c{args.other_chars}" if args.other_chars != 1400 else ""))


def answers(args) -> None:
    import psutil
    from diag.agent import Agent
    proc = psutil.Process()
    t = time.perf_counter()
    agent = Agent(model=args.model, n_threads=args.threads, k=args.k, other_chars=args.other_chars)
    load_s = time.perf_counter() - t
    rss_loaded = proc.memory_info().rss
    qs = forum_set() + manual_set(args.n)
    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"answers_{Path(args.model).stem}{variant(args)}.jsonl"
    rows, peak = [], rss_loaded
    with open(out_path, "w", encoding="utf-8") as fh:
        for q in qs:
            a = agent.ask(q["question"], device=q["device"], closed_book=args.closed_book)
            peak = max(peak, proc.memory_info().rss)
            row = {**check(q, a.text, a.mentioned, a.unsupported, a.invented, a.ack_conflict), "set": "manual" if "entry" in q else "forum", "answer": a.text,
                   "sources": a.sources, "retrieve_ms": a.retrieve_ms, "generate_ms": a.generate_ms,
                   "prompt_tokens": a.prompt_tokens, "completion_tokens": a.completion_tokens}
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(q["id"], "grounded" if a.grounded else f"NOT grounded {a.unsupported} {a.invented}",
                  f"{a.generate_ms / 1000:.1f}s", flush=True)
    lat = sorted(r["generate_ms"] + r["retrieve_ms"] for r in rows)
    tps = [r["completion_tokens"] / (r["generate_ms"] / 1000) for r in rows if r["generate_ms"]]
    summary = {
        "model": Path(args.model).name, "closed_book": args.closed_book, "threads": args.threads,
        "k": args.k, "other_chars": args.other_chars,
        "load_s": round(load_s, 1), "rss_after_load_mb": round(rss_loaded / 2**20), "rss_peak_mb": round(peak / 2**20),
        "latency_p50_s": round(statistics.median(lat) / 1000, 1), "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))] / 1000, 1),
        "completion_tokens_per_s_median": round(statistics.median(tps), 1),
    }
    summary.update(quality(rows))
    json.dump(summary, open(OUT / f"summary_{out_path.stem.removeprefix('answers_')}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


def rescore(args) -> None:
    from diag.agent import ack_conflicts, facts, ground
    from diag.retrieve import Doc, load_docs
    docs = {(d.id, d.device): d for d in load_docs(KB)}
    known = {d.id for d in docs.values()}
    qs = {q["id"]: q for q in forum_set() + manual_set(args.n)}
    for path in sorted(OUT.glob("answers_*.jsonl")):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        spath = OUT / f"summary_{path.stem.removeprefix('answers_')}.json"
        old = json.load(open(spath))
        for i, r in enumerate(rows):
            q = qs[r["id"]]
            shown = [Doc(**{**docs[(s["id"], s["device"])].__dict__, "why": s["why"]}) for s in r["sources"]]
            mentioned, unsupported, invented = ground(r["answer"], q["question"], shown, known, old.get("other_chars", 1400))
            conflict = ack_conflicts(r["answer"], facts(shown))
            keep = {k: r[k] for k in ("set", "answer", "sources", "retrieve_ms", "generate_ms", "prompt_tokens", "completion_tokens")}
            rows[i] = {**check(q, r["answer"], mentioned, unsupported, invented, conflict), **keep}
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        summary = {k: old[k] for k in ("model", "closed_book", "threads", "k", "other_chars", "load_s", "rss_after_load_mb",
                                       "rss_peak_mb", "latency_p50_s", "latency_p95_s", "completion_tokens_per_s_median")
                   if k in old}
        summary.update(quality(rows))
        json.dump(summary, open(spath, "w"), indent=1)
        print(path.name, json.dumps(summary, indent=1))


_FORUM = None


def forum_gold(qid: str) -> list:
    global _FORUM
    _FORUM = _FORUM or {q["id"]: q["gold"] for q in forum_set()}
    return _FORUM.get(qid, [])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "answers", "rescore"])
    ap.add_argument("--model", default="data/models/qwen2.5-3b-instruct-q4_k_m.gguf")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--closed-book", action="store_true")
    ap.add_argument("--k", type=int, default=5, help="manual entries retrieved per question")
    ap.add_argument("--other-chars", type=int, default=1400, help="excerpt length for all but the exactly matched entry")
    a = ap.parse_args()
    {"retrieval": retrieval, "answers": answers, "rescore": rescore}[a.what](a)
