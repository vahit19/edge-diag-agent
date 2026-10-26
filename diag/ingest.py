"""Parse SINAMICS List Manuals into structured fault/alarm and parameter records.

    python -m diag.ingest data/kb data/raw/g120_list_manual_0223.pdf data/raw/g120c_list_manual_0223.pdf

Writes data/kb/messages.jsonl (one record per fault/alarm, with message class, reaction,
acknowledgment, cause and remedy exactly as printed) and data/kb/parameters.jsonl (one
record per parameter). Page numbers are kept so every answer can cite the manual page.
"""

import json
import re
import sys
from pathlib import Path

import pymupdf

CODE_LINE = re.compile(r"^([FAN]\d{5})(?: \(([FAN, ]+)\))?$")
HEADER_PREFIXES = ("SINAMICS G120 Control Units", "SINAMICS G120C", "List Manual, ", "4 Faults and alarms", "4.2 List of faults and alarms",
                   "2 Parameters", "2.2 List of parameters")


def page_lines(doc: pymupdf.Document, first: int, last: int):
    """Yield (page_number, line) for 1-based pages first..last, without running headers."""
    for pno in range(first - 1, last):
        for line in doc[pno].get_text().splitlines():
            line = line.strip()
            if not line or line.startswith(HEADER_PREFIXES) or re.fullmatch(r"\d{1,3}", line):
                continue
            yield pno + 1, line


def chapter_pages(doc: pymupdf.Document, title: str) -> tuple:
    toc = doc.get_toc()
    for i, (lvl, t, p) in enumerate(toc):
        if t.startswith(title):
            nxt = next(q for l2, _, q in toc[i + 1:] if l2 <= lvl)
            return p, nxt - 1
    raise ValueError(title)


def parse_messages(doc: pymupdf.Document) -> list:
    first, last = chapter_pages(doc, "4.2 List of faults and alarms")
    records, cur, field = [], None, None
    lines = list(page_lines(doc, first, last))
    i = 0
    while i < len(lines):
        page, line = lines[i]
        m = CODE_LINE.match(line)
        if m and i + 1 < len(lines) and lines[i + 1][1] != "Cause":
            if cur:
                records.append(cur)
            cur = {"code": m.group(1), "type": {"F": "fault", "A": "alarm", "N": "no message"}[m.group(1)[0]],
                   "can_change_to": m.group(2) or "", "title": lines[i + 1][1], "page": page,
                   "message_class": "", "reaction": "", "acknowledge": "", "cause": "", "remedy": ""}
            field = None
            i += 2
            continue
        if cur is not None:
            key = {"Message class:": "message_class", "Reaction:": "reaction", "Acknowledge:": "acknowledge",
                   "Cause": "cause", "Remedy": "remedy"}.get(line)
            if key:
                field = key
            elif field:
                cur[field] = (cur[field] + "\n" + line).strip()
        i += 1
    if cur:
        records.append(cur)
    return records


def parse_parameters(doc: pymupdf.Document) -> list:
    """Parameter headers sit in table cells that plain text extraction drops, so each
    parameter's text is cut from the page region between its bookmark and the next one."""
    toc = doc.get_toc(simple=False)
    start = next(i for i, e in enumerate(toc) if e[1].startswith("2.2 List of parameters"))
    end = next(i for i, e in enumerate(toc) if e[1].startswith("2.3 "))
    marks = [(e[2], doc[e[2] - 1].rect.height - e[3]["to"].y, e[1]) for e in toc[start + 1:end + 1]]
    records = []
    for (page, y, title), (npage, ny, _) in zip(marks, marks[1:]):
        m = re.match(r"^([pr]\d{4,5})\S*\s+(.+?)(?: / (.+))?$", title)
        parts = []
        for pno in range(page, npage + 1):
            for x0, y0, x1, y1, text, *_ in doc[pno - 1].get_text("blocks", sort=True):
                if (pno == page and y0 < y - 2) or (pno == npage and y0 >= ny - 2):
                    continue
                text = text.strip()
                if text and not text.startswith(HEADER_PREFIXES) and not re.fullmatch(r"\d{1,3}", text):
                    parts.append(text)
        if records and records[-1]["number"] == m.group(1):  # variants of one parameter
            records[-1]["text"] += "\n\n" + "\n".join(parts)
            continue
        records.append({"number": m.group(1), "name": m.group(2), "short": m.group(3) or "", "page": page,
                        "text": "\n".join(parts)})
    return records


def main() -> None:
    out, pdfs = Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]]
    out.mkdir(parents=True, exist_ok=True)
    collected = {"messages": [], "parameters": []}
    for pdf in pdfs:
        doc = pymupdf.open(pdf)
        device = doc.metadata.get("title") or pdf.stem
        for name, recs in (("messages", parse_messages(doc)), ("parameters", parse_parameters(doc))):
            for r in recs:
                r["source"], r["device"] = pdf.name, device
            collected[name] += recs
            print(pdf.name, name, len(recs))
    for name, recs in collected.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
