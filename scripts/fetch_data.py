"""Download the public Siemens List Manuals and the quantized models, then build the knowledge base.

    python scripts/fetch_data.py

The manuals are Siemens documents; they are downloaded from Siemens' own servers and are not
redistributed in this repository.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

MANUALS = {
    "g120_list_manual_0223.pdf": "https://cache.industry.siemens.com/dl/files/919/109817919/att_1133702/v1/G120_CU240BE-2_list_man_0223_en-US.pdf",
    "g120c_list_manual_0223.pdf": "https://cache.industry.siemens.com/dl/files/922/109817922/att_1133717/v1/G120C_list_man_0223_en-US.pdf",
}
MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
]


def main() -> None:
    raw, models = Path("data/raw"), Path("data/models")
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in MANUALS.items():
        if not (raw / name).exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            (raw / name).write_bytes(urllib.request.urlopen(req).read())
        print("manual", name, (raw / name).stat().st_size)
    from huggingface_hub import hf_hub_download
    for repo, f in MODELS:
        print("model", hf_hub_download(repo, f, local_dir=models))
    subprocess.run([sys.executable, "-m", "diag.ingest", "data/kb", *[str(raw / n) for n in MANUALS]], check=True)


if __name__ == "__main__":
    main()
