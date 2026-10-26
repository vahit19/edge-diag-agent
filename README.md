# edge-diag-agent

An offline diagnostic assistant for Siemens SINAMICS G120 / G120C drives that runs on a few CPU
cores with no internet connection. A technician types what the drive shows, for example
"G120C shows F7900 when the conveyor starts in automatic mode". The assistant then:

1. finds the matching entries in the official List Manuals stored on the device;
2. answers with a small quantized language model (Qwen2.5 3B or 1.5B, 4-bit GGUF, llama.cpp);
3. checks the answer before showing it.

Each answer comes with manual pages the technician can open. The check flags any fault code or
parameter number the model was not shown. The fault's reaction and acknowledgment are copied
word for word from the manual and shown beside the model's text, never paraphrased.

Everything is measured end to end on real data: retrieval quality, answer correctness, latency
and memory. The numbers below come from `results/`, produced by `eval/run_eval.py`.

## Headline results

On 40 fault/alarm entries sampled from the manuals (seed 0), the model was asked
"The drive shows `<code>`. What does it mean and what should I do?":

| | 3B + manuals (default) | 1.5B + manuals | 3B, no manuals |
|---|---|---|---|
| Meaning matches the manual title (mean word overlap) | **0.94** | 0.83 | 0.06 |
| Fault acknowledgment stated correctly (24 faults) | **21/24** (88%, 95% CI 69–96) | 19/24 (79%, 60–91) | 0/24 |
| Fault acknowledgment stated wrongly | 2/24 (8%, 2–26) | 0/24 (0–14) | 0/24 |
| Answers naming a code or parameter it was not shown | 0/40 (0–9) | 1/40 (0–13) | n/a |
| Latency per answer, p50 / p95 (4 threads) | 32.9 s / 55.1 s | 19.2 s / 31.7 s | 12.4 s / 22.2 s |
| Peak memory (process RSS) | 3.9 GB | 2.2 GB | 3.7 GB |

Without the manuals, the same 3B model attributed 30 of the 40 codes to an encoder problem
within its first few sentences. None of the 40 entries is an encoder fault. With the manuals it
gets the meaning right, but in 2 of 24 faults it copied the line "Reaction: NONE" as the
acknowledgment. The manual says "Acknowledge: POWER ON" for both.

The acknowledgment gate catches this kind of error: it compares the mode stated in the answer
with the manual entry. On these 40 answers it flagged both wrong ones (2/2) and none of the
other 38 (0/38). The gate was written after seeing these two errors, so this result shows the
gate works as designed. It is not an independent test. In the web page, the manual's own line
is always shown above the answer either way.

## Real questions from technicians

The forum set has 9 real questions from public PLCtalk threads about G120/G120C drives. The gold
answer for each is the code(s) and parameter(s) named in the thread's resolution, checked
against the manual. The question texts are kept out of this repository; the threads are:

| Thread | Gold |
|---|---|
| [G120C DP 1910 fault](https://www.plctalk.net/forums/threads/g120c-dp-1910-fault.134179/) | F01910, p2040 |
| [Inverter Sinamics G120C](https://www.plctalk.net/forums/threads/inverter-sinamics-g120c.144942/) | F07807, p1901 |
| [G120C not running automatically](https://www.plctalk.net/forums/threads/g120c-not-running-automatically.135377/) | F07900, p1900 |
| [Sinamics G120 drive](https://www.plctalk.net/forums/threads/sinamics-g120-drive.119012/) | F30002, p1121 |
| [Built-in motor PTC with Siemens G120 VFD](https://www.plctalk.net/forums/threads/built-in-motor-ptc-with-siemens-g120-vfd.96052/) | r0035, p0601 |
| [How to reset G120 fault over Profinet](https://www.plctalk.net/forums/threads/how-to-reset-g120-fault-over-profinet.90356/) | p2103, r0054 |
| [TIA Portal Sinamics G120C doesn't start](https://www.plctalk.net/forums/threads/tia-portal-sinamics-g120c-doesnt-start.111058/) | r0046, r0054 |
| [Siemens G120 VSD flying start](https://www.plctalk.net/forums/threads/siemens-g120-vsd-flying-start.106779/) | A07409, A07910, F07900 |
| [Sinamics G120 VFD DI problem](https://www.plctalk.net/forums/threads/sinamics-g120-vfd-di-problem.96025/) | none: a wiring question the List Manual does not cover |

### Retrieval

The table counts gold items in the top 5 entries plus the linked parameters: 8 questions with
gold, 17 gold items.

| Configuration | Gold items found | Questions fully covered |
|---|---|---|
| Full: exact identifiers + BM25 + dense + linked parameters | **11/17** (65%, 95% CI 41–83) | 4/8 |
| Without linked parameters | 9/17 (53%) | 2/8 |
| Without exact identifier lookup | 4/17 (24%) | 1/8 |
| BM25 only | 4/17 (24%) | 2/8 |
| Dense embeddings only | 3/17 (18%) | 1/8 |

The biggest lever is plain code: technicians write "F7900", "O.F 7807" or "the 1910 error", and
normalising these to manual codes (F07900, F07807, F01910) before any fuzzy search takes
coverage from 24% to 65%. Retrieval takes about 30 ms.

### Answers

On the forum questions, all 9 answers from both models mentioned only codes and parameters the
model had been shown. The answers named 9/17 gold items with the 3B model and 8/17 with 1.5B,
limited by what retrieval found. Neither model declined the out-of-scope wiring question. The
page shows a warning when a question names no code found in the manuals, but one question is
too few to measure whether that is enough.

## How it works

```
question ──► identifier normaliser ──► exact lookup ─┐
         └─► BM25 ───────────────┐                    ├─► top 5 + parameters the matched
         └─► bge-small (ONNX) ───┴─► rank fusion ─────┘   entry's Remedy points to
                                                                │
                      manual excerpts tagged [S1]..[Sn] ◄───────┘
                                     │
                      Qwen2.5 3B Q4_K_M (llama.cpp, CPU)
                                     │
             checks: every code/parameter in the answer was shown to the model;
             stated acknowledgment agrees with the manual entry
                                     │
             web page: manual facts verbatim, answer, warnings, manual + page for each source
```

| File | What it does |
|---|---|
| `diag/ingest.py` | Parses the List Manual PDFs: 661 fault/alarm entries (code, reaction, acknowledgment, cause, remedy) and 3028 parameter descriptions, cut by the PDF's own bookmark coordinates |
| `diag/retrieve.py` | Identifier normalisation, BM25, dense vectors, reciprocal-rank fusion, device filter, linked parameters |
| `diag/agent.py` | Prompt, local generation, grounding check, acknowledgment check |
| `diag/server.py` | FastAPI: `POST /api/ask`, `GET /api/health`, a one-page UI; one generation at a time |
| `eval/run_eval.py` | `retrieval`, `answers`, `rescore` modes; Wilson 95% intervals |
| `tests/` | Checks that need neither the manuals nor a model |

Knowledge base: the SINAMICS G120 Control Units CU240B-2/CU240E-2 List Manual (A5E33839529,
02/2023) and the SINAMICS G120C List Manual (A5E33840768, 02/2023). `scripts/fetch_data.py`
downloads both from Siemens' servers. They are not redistributed here.

## Run it

```bash
python -m venv .venv && . .venv/Scripts/activate        # or bin/activate
pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt
python scripts/fetch_data.py                             # manuals, two GGUF models, parse
python -m diag.agent "G120C shows F7900 when running in automatic mode"
uvicorn diag.server:app --host 0.0.0.0 --port 8080       # http://localhost:8080
python -m eval.run_eval retrieval
python -m eval.run_eval answers --model data/models/qwen2.5-3b-instruct-q4_k_m.gguf --n 40
pip install pytest && pytest -q tests
```

The forum evaluation needs `data/eval/forum_questions.jsonl`. Each line has `id`, `device`,
`question` and `gold`. Write the questions from the threads above.

### Container

The `Dockerfile` bakes everything the device needs into the image at build time: the parsed
manuals, the embedding index and one model. At run time it has no network access
(`HF_HUB_OFFLINE=1`) and runs as a non-root user. `docker-compose.yml` follows the shape
Industrial Edge apps use: one service, a fixed tag, an explicit memory limit and a restart
policy. With the 3B model use 5 GB of memory; for a 3 GB budget, build with
`MODEL=qwen2.5-1.5b-instruct-q4_k_m.gguf`.

## Limitations

- **Small sets.** 9 forum questions, 40 manual entries. The confidence intervals are wide and
  are reported with every number.
- **The forum set was also used for tuning.** Linked-parameter expansion was added after looking
  at its misses. A held-out set of new threads is needed before quoting the retrieval number as
  a generalisation estimate.
- **The manual set is easier than real questions.** It asks for one code by name, so it tests
  reading and reporting rather than diagnosis.
- **Simple scoring.** Meaning is scored by word overlap with the manual title, and
  acknowledgment by the mode stated next to the word "acknowledge". Both are rough checks;
  answers were also read by hand for the failure cases above.
- **Not measured on industrial hardware.** Timings are from a laptop CPU (Intel i7-13800H)
  limited to 4 threads. An industrial PC with fewer or slower cores will be slower. The
  container image has not yet been built and timed on an Industrial Edge device.
- **Questions without a code** are the weak case, for example "47F, speed controller
  inhibited". Retrieval misses the status parameters they need (r0046, r0054). The List
  Manual also lacks wiring and commissioning steps; the operating instructions, which cover
  them, are not in the knowledge base.
