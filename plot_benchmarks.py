"""Read JSONs in --input dir, produce 2 charts + 1 summary table.

Charts:
  headline.png    — capability vs cost, with arrow showing distillation lift
  throughput.png  — throughput vs batch size
Table:
  summary.md      — all metrics, all models, paste into slide
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#4e47c2"
BLACK = "#000000"
WHITE = "#ffffff"
INFEASIBLE = "#f0f0f0"

ORDER = ["teacher", "base_student", "distilled"]
LABELS = {
    "teacher": "Teacher 7B",
    "base_student": "Base 1.5B",
    "distilled": "Distilled 1.5B",
}
COLORS = {"teacher": WHITE, "base_student": BLACK, "distilled": ACCENT}
EDGES = {"teacher": BLACK, "base_student": BLACK, "distilled": ACCENT}

# Approximate VRAM for teacher in bf16 — placed in the infeasible region on the headline plot.
TEACHER_VRAM_ESTIMATE_GB = 14.0
VRAM_CAP_GB = 8.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,
    "axes.titlecolor": BLACK,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "text.color": BLACK,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})


def load_records(input_dir: Path):
    recs = {}
    for p in input_dir.glob("*.json"):
        with open(p) as f:
            r = json.load(f)
        recs[r["name"]] = r
    teacher_path = Path("teacher.json")
    if teacher_path.exists() and "teacher" not in recs:
        with open(teacher_path) as f:
            recs["teacher"] = json.load(f)
    return [recs[n] for n in ORDER if n in recs]


# ─── helpers ──────────────────────────────────────────────────────────────────

def get_accuracy(r):
    return (r.get("accuracy") or {}).get("accuracy")


def get_vram(r):
    return (r.get("vram") or {}).get("peak_during_gsm8k_gb")


def get_latency_p50(r):
    return (r.get("latency") or {}).get("p50_per_token_ms")


def get_throughput_at(r, bs):
    for t in (r.get("throughput") or []):
        if t.get("batch_size") == bs and not t.get("oom"):
            return t.get("tokens_per_s")
    return None


def get_max_throughput(r):
    valid = [t["tokens_per_s"] for t in (r.get("throughput") or []) if not t.get("oom")]
    return max(valid) if valid else None


def get_jpc(r):
    return r.get("joules_per_correct")


def fmt(v, suffix="", precision=1, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{precision}f}{suffix}"
    return f"{v}{suffix}"


# ─── charts ───────────────────────────────────────────────────────────────────

def chart_headline(recs, out: Path):
    """Capability vs cost. The story in one image."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

    points = {}  # name -> (x, y, plotted)
    for r in recs:
        name = r["name"]
        acc = get_accuracy(r)
        if acc is None:
            continue
        if name == "teacher":
            x = TEACHER_VRAM_ESTIMATE_GB
        else:
            x = get_vram(r)
            if x is None:
                continue
        y = acc * 100
        points[name] = (x, y)

    # Shade infeasible region beyond the cap
    xmax = 17
    ax.axvspan(VRAM_CAP_GB, xmax, color=INFEASIBLE, zorder=0)
    ax.axvline(VRAM_CAP_GB, color=ACCENT, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(VRAM_CAP_GB + 0.15, 5, f"{VRAM_CAP_GB:.0f} GB cap",
            color=ACCENT, fontsize=10, va="bottom", fontweight="bold")
    ax.text((VRAM_CAP_GB + xmax) / 2, 12, "infeasible on 8 GB hardware",
            color="#888", fontsize=10, ha="center", style="italic")

    # Arrow from base → distilled, showing the lift from distillation
    if "base_student" in points and "distilled" in points:
        bx, by = points["base_student"]
        dx, dy = points["distilled"]
        ax.annotate(
            "",
            xy=(dx, dy), xytext=(bx, by),
            arrowprops=dict(
                arrowstyle="->", color=ACCENT, lw=2.2,
                shrinkA=14, shrinkB=14,
            ),
            zorder=2,
        )
        mid_x, mid_y = (bx + dx) / 2, (by + dy) / 2
        delta = dy - by
        ax.text(
            mid_x, mid_y + 4,
            f"+{delta:.1f} pts\nfrom distillation",
            color=ACCENT, fontsize=10, fontweight="bold",
            ha="center", va="bottom",
        )

    # Plot each model marker
    for name, (x, y) in points.items():
        ax.scatter(
            x, y, s=420,
            c=COLORS[name], edgecolors=EDGES[name],
            linewidths=2.2, zorder=3,
        )
        # Label placement: teacher to the left (since it's near the right edge), others to the right
        if name == "teacher":
            ax.annotate(
                f"{LABELS[name]}\n{y:.1f}%",
                xy=(x, y), xytext=(-14, 0), textcoords="offset points",
                ha="right", va="center", fontsize=11, fontweight="bold",
            )
        else:
            ax.annotate(
                f"{LABELS[name]}\n{y:.1f}%",
                xy=(x, y), xytext=(14, 0), textcoords="offset points",
                ha="left", va="center", fontsize=11, fontweight="bold",
            )

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Peak VRAM during inference (GB)")
    ax.set_ylabel("GSM8K Accuracy (%)")
    ax.set_title("Distilled student keeps teacher-level accuracy\nat a fraction of the memory cost",
                 fontweight="bold", pad=12)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "headline.png", dpi=150)
    plt.close(fig)


def chart_throughput(recs, out: Path):
    """Single line chart: throughput vs batch size."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in recs:
        bs_list = [t["batch_size"] for t in (r.get("throughput") or []) if not t.get("oom")]
        tps_list = [t["tokens_per_s"] for t in (r.get("throughput") or []) if not t.get("oom")]
        if not bs_list:
            continue
        line_color = ACCENT if r["name"] == "distilled" else BLACK
        ax.plot(bs_list, tps_list, marker="o", label=LABELS[r["name"]],
                color=line_color, linewidth=2.5, markersize=9)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput as concurrency grows", fontweight="bold", pad=12)
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=150)
    plt.close(fig)


# ─── summary table ────────────────────────────────────────────────────────────

def write_summary(recs, out: Path):
    """Markdown table with everything that didn't earn a chart."""
    by_name = {r["name"]: r for r in recs}
    cols = [n for n in ORDER if n in by_name]
    headers = [LABELS[c] for c in cols]

    def row(label, getter, fmt_fn=lambda v: fmt(v)):
        return [label] + [fmt_fn(getter(by_name[c])) for c in cols]

    rows = [
        row("GSM8K accuracy", lambda r: get_accuracy(r),
            lambda v: fmt(v * 100, "%", precision=1) if v is not None else "—"),
        row("Peak VRAM", lambda r: get_vram(r),
            lambda v: fmt(v, " GB", precision=2) if v is not None else "OOM (>8 GB)"),
        row("Latency p50", lambda r: get_latency_p50(r),
            lambda v: fmt(v, " ms/tok", precision=1)),
        row("Throughput @ bs=1", lambda r: get_throughput_at(r, 1),
            lambda v: fmt(v, " tok/s", precision=0)),
        row("Throughput @ bs=8", lambda r: get_throughput_at(r, 8),
            lambda v: fmt(v, " tok/s", precision=0)),
        row("Max throughput", lambda r: get_max_throughput(r),
            lambda v: fmt(v, " tok/s", precision=0)),
        row("Energy / correct", lambda r: get_jpc(r),
            lambda v: fmt(v, " J", precision=0)),
    ]

    lines = []
    lines.append("# Benchmark Summary\n")
    lines.append("| Metric | " + " | ".join(headers) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")

    notes = []
    teacher = by_name.get("teacher")
    if teacher and (teacher.get("vram") or {}).get("oom_on_8gb"):
        notes.append(f"- Teacher 7B does not load under the {VRAM_CAP_GB:.0f} GB cap; "
                     f"speed and memory metrics are N/A. Accuracy is cited from prior measurement.")
    notes.append("- All measurements use greedy decode (`do_sample=False`) for reproducibility.")
    notes.append("- Energy = total joules sampled at 10 Hz over the GSM8K eval, divided by # correct answers.")

    if notes:
        lines.append("\n**Notes**\n")
        lines.extend(notes)

    out_path = out / "summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="bench_results")
    p.add_argument("--output", default="bench_results/charts")
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_records(in_dir)
    if not recs:
        print(f"No JSONs found in {in_dir}")
        return

    chart_headline(recs, out_dir)
    chart_throughput(recs, out_dir)
    write_summary(recs, out_dir)
    print(f"Wrote 2 charts + summary.md to {out_dir}")


if __name__ == "__main__":
    main()
