"""Draw the README figures from results/*.json, in a light and a dark version.

    pip install matplotlib && python -m eval.figures   # writes docs/figures/<name>.svg and <name>-dark.svg

Every bar comes from the saved summaries; nothing is typed in by hand.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from matplotlib.transforms import IdentityTransform  # noqa: E402

RES, OUT = Path("results"), Path("docs/figures")
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", quiet="#a8a6a0"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s1="#3987e5", s2="#d95926", quiet="#6b6a65"),
}
BAR, R = 13, 4  # bar thickness and rounded data-end radius, points (72 dpi: 1 pt = 1 SVG unit)


def summary(name: str) -> dict:
    return json.load(open(RES / f"summary_qwen2.5-{name}.json"))


def frac(s: str) -> tuple:
    """'21/24 (88%, 95% CI 69-96)' -> (21, 24)."""
    k, n = re.match(r"(\d+)/(\d+)", s).groups()
    return int(k), int(n)


def _setup(t: dict) -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 10.5,
        "svg.hashsalt": "edgediag", "svg.fonttype": "path", "figure.dpi": 72, "savefig.dpi": 72,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["muted"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 1, "xtick.major.size": 0, "ytick.major.size": 0,
    })


def _chrome(ax, t: dict) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.grid(axis="x", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def _title(fig, title: str, sub: str, t: dict) -> None:
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=14, weight="semibold", color=t["ink"])
    fig.text(0.02, 0.965 - 0.45 / fig.get_figheight(), sub, ha="left", va="top", fontsize=10.5, color=t["ink2"])


def _bar(ax, x0: float, x1: float, y: float, color: str) -> None:
    """Horizontal bar, square at the baseline, rounded data end; drawn in display space."""
    (a, yc), (b, _) = ax.transData.transform([(x0, y), (x1, y)])
    if abs(b - a) < 0.5:
        return
    s = 1 if b > a else -1
    h, r = BAR / 2, min(R, abs(b - a), BAR / 2)
    th = np.linspace(0, np.pi / 2, 8)
    top = [(b - s * r + s * r * np.sin(q), yc + h - r + r * np.cos(q)) for q in th]
    bot = [(b - s * r + s * r * np.cos(q), yc - h + r - r * np.sin(q)) for q in th]
    ax.add_patch(Polygon([(a, yc + h)] + top + bot + [(a, yc - h)], closed=True, transform=IdentityTransform(),
                         facecolor=color, edgecolor="none", zorder=3))


def _legend(fig, items: list, t: dict, y: float) -> None:
    x = 0.02
    for label, color in items:
        fig.patches.append(plt.Rectangle((x, y - 0.012), 0.014, 0.024, transform=fig.transFigure,
                                         facecolor=color, edgecolor="none"))
        txt = fig.text(x + 0.022, y, label, va="center", fontsize=10, color=t["ink2"])
        fig.canvas.draw()
        x = fig.transFigure.inverted().transform(txt.get_window_extent().corners()[-1])[0] + 0.03


def manuals(t: dict, path: Path) -> None:
    """Hero: the same 3B model with and without the manuals on the device."""
    closed, rag = summary("3b-instruct-q4_k_m_closedbook"), summary("3b-instruct-q4_k_m")
    rows = []
    for label, key in [("Names an encoder problem\n(first 300 characters)", "manual_encoder_in_first_300_chars"),
                       ("States the fault's acknowledgment\ncorrectly (24 faults)", "manual_fault_ack_correct")]:
        rows.append((label, [frac(closed[key]), frac(rag[key])]))
    o_closed, o_rag = closed["manual_meaning_overlap_mean"], rag["manual_meaning_overlap_mean"]
    fig = plt.figure(figsize=(9.6, 4.6))
    ax = fig.add_axes([0.3, 0.12, 0.66, 0.56])
    _chrome(ax, t)
    ax.set_xlim(0, 100)
    labels = [r[0] for r in rows] + ["Meaning matches the manual\n(mean word overlap with title)"]
    ax.set_ylim(len(labels) - 0.45, -0.6)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    fig.canvas.draw()
    off = 0.2
    vals = [[(k / n * 100, f"{k}/{n}") for k, n in r[1]] for r in rows]
    vals.append([(o_closed * 100, f"{o_closed:.2f}"), (o_rag * 100, f"{o_rag:.2f}")])
    for i, ((v0, l0), (v1, l1)) in enumerate(vals):
        for v, lab, y, c in ((v0, l0, i - off, t["s2"]), (v1, l1, i + off, t["s1"])):
            _bar(ax, 0, v, y, c)
            ax.text(v + 1.2, y, lab, va="center", fontsize=10, color=t["ink"])
    _title(fig, "Without the manuals, the model guesses \"encoder\"",
           "Qwen2.5 3B (4-bit) on 40 fault/alarm codes from the G120/G120C List Manuals; none is an encoder fault", t)
    _legend(fig, [("No manuals", t["s2"]), ("Manuals on the device", t["s1"])], t, 0.77)
    fig.savefig(path)
    plt.close(fig)


def retrieval(t: dict, path: Path) -> None:
    rows = json.load(open(RES / "retrieval.json"))
    names = {"full (exact + BM25 + dense + linked params)": "Full: exact codes + BM25 + dense + linked",
             "no linked parameters": "Without linked parameters", "no exact identifier lookup": "Without exact code lookup",
             "BM25 only": "BM25 only", "dense only": "Dense embeddings only"}
    fig = plt.figure(figsize=(9.6, 4.2))
    ax = fig.add_axes([0.3, 0.12, 0.66, 0.6])
    _chrome(ax, t)
    n = rows[0]["gold_items"]
    ax.set_xlim(0, n)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_yticks(range(len(rows)), [names[r["config"]] for r in rows])
    ax.set_xticks(range(0, n + 1, 4))
    ax.set_xlabel(f"gold codes and parameters found, of {n} (8 forum questions)", color=t["muted"], fontsize=9.5)
    fig.canvas.draw()
    for i, r in enumerate(rows):
        _bar(ax, 0, r["gold_items_found"], i, t["s1"] if i == 0 else t["quiet"])
        ax.text(r["gold_items_found"] + 0.2, i, f"{r['gold_items_found']}/{n}", va="center", fontsize=10, color=t["ink"])
    _title(fig, "Reading \"F7900\" as F07900 matters more than the search model",
           "Real technician questions from PLCtalk; top 5 entries plus the parameters their Remedy points to", t)
    fig.savefig(path)
    plt.close(fig)


def latency(t: dict, path: Path) -> None:
    vm = json.load(open(RES / "bench_vlab_playground_1.5b.json"))["summary"]
    rows = []
    for label, name in [("Laptop, 4 threads", "1.5b-instruct-q4_k_m"), ("Laptop, 2 threads", "1.5b-instruct-q4_k_m_t2"),
                        ("Laptop, 2 threads, short excerpts", "1.5b-instruct-q4_k_m_t2_k3_c400")]:
        s = summary(name)
        rows.append((label, s["latency_p50_s"], s["latency_p95_s"], t["s1"] if "short" in label else t["quiet"]))
    rows.append(("Playground VM, 2 cores", vm["total_ms_p50"] / 1000, vm["total_ms_p95"] / 1000, t["s2"]))
    fig = plt.figure(figsize=(9.6, 4.0))
    ax = fig.add_axes([0.3, 0.14, 0.66, 0.56])
    _chrome(ax, t)
    ax.set_xlim(0, 360)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xticks(range(0, 301, 60), [f"{v} s" for v in range(0, 301, 60)])
    ax.set_xlabel("seconds per answer: bar = median, tick = 95th percentile", color=t["muted"], fontsize=9.5)
    fig.canvas.draw()
    for i, (_, p50, p95, c) in enumerate(rows):
        _bar(ax, 0, p50, i, c)
        ax.plot([p95, p95], [i - 0.25, i + 0.25], color=t["ink2"], linewidth=2, solid_capstyle="round", zorder=4)
        ax.text(p95 + 4, i, f"{p50:.0f} s / {p95:.0f} s", va="center", fontsize=10, color=t["ink"])
    _title(fig, "Two cores need a shorter prompt",
           "Qwen2.5 1.5B (4-bit); short = 3 entries, 400 characters each except the matched one", t)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme, t in THEMES.items():
        _setup(t)
        suffix = "" if theme == "light" else "-dark"
        for draw in (manuals, retrieval, latency):
            draw(t, OUT / f"{draw.__name__}{suffix}.svg")
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
