"""Time the running service on the target device, with nothing but the Python standard library.

    python scripts/bench_api.py --url http://<device>:8080 --out bench_device.json [--n 40]

Asks the same 40 manual questions as `eval/run_eval.py answers` (seed 0), one at a time, and
reports end-to-end latency, the server's own retrieve/generate split and how many answers
passed both checks. Memory is read on the device itself, e.g. `docker stats --no-stream`.
"""

import argparse
import json
import statistics
import time
import urllib.request

G120C, G120 = "SINAMICS G120C", "SINAMICS G120 CU240B-2/CU240E-2"
# (code, device) of the 40 entries `manual_set(40, seed=0)` picks, in the same order.
MANUAL_SET = [
    ("F01152", G120C), ("A01788", G120C), ("F01122", G120), ("F30011", G120), ("A07892", G120C),
    ("A07352", G120C), ("F01650", G120C), ("F30659", G120), ("A07094", G120C), ("F01023", G120C),
    ("F30036", G120C), ("F07967", G120), ("F07802", G120C), ("A06921", G120), ("A30054", G120),
    ("F06922", G120), ("A01788", G120), ("F30805", G120C), ("F13009", G120), ("F07985", G120C),
    ("A30502", G120C), ("F07082", G120), ("F30682", G120), ("A01799", G120), ("A01666", G120),
    ("F30903", G120), ("F07084", G120C), ("F13101", G120C), ("F01910", G120), ("F01018", G120C),
    ("A02057", G120C), ("A30706", G120), ("F30659", G120C), ("A07920", G120), ("A08755", G120C),
    ("F02080", G120C), ("A07929", G120C), ("F30012", G120), ("F01651", G120), ("A08751", G120C),
]


def call(url: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def pct(xs: list, q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=len(MANUAL_SET))
    ap.add_argument("--out", default="bench_device.json")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    health = call(base + "/api/health", timeout=30)
    print("health:", health)

    rows = []
    for code, device in MANUAL_SET[:args.n]:
        q = f"The drive ({device}) shows {code}. What does it mean and what should I do?"
        t = time.perf_counter()
        r = call(base + "/api/ask", {"question": q, "device": device})
        total = (time.perf_counter() - t) * 1000
        rows.append({"code": code, "device": device, "total_ms": round(total), **r["timing_ms"],
                     "passed": r["passed"], "ack_conflict": r["ack_conflict"], "unsupported": r["unsupported"],
                     "invented": r["invented"]})
        print(f"{code:7} {device[-14:]:>14}  {total / 1000:6.1f} s  passed={r['passed']}")

    total = [r["total_ms"] for r in rows]
    summary = {"url": base, "health": health, "n": len(rows),
               "total_ms_p50": round(statistics.median(total)), "total_ms_p95": pct(total, 0.95),
               "generate_ms_p50": round(statistics.median(r["generate"] for r in rows)),
               "retrieve_ms_p50": round(statistics.median(r["retrieve"] for r in rows)),
               "passed": sum(r["passed"] for r in rows), "ack_conflicts": sum(bool(r["ack_conflict"]) for r in rows)}
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
