"""Read all JSONs in --input dir and emit charts to --output.

Skips charts where a model has null values (e.g. teacher OOM has no speed numbers).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ORDER = ["teacher", "base_student", "distilled"]
LABELS = {"teacher": "Teacher (7B)", "base_student": "Base Student (1.5B)", "distilled": "Distilled (1.5B)"}
COLORS = {"teacher": "#888888", "base_student": "#4a90e2", "distilled": "#e2724a"}


def load_records(input_dir: Path):
    recs = {}
    for p in input_dir.glob("*.json"):
        with open(p) as f:
            r = json.load(f)
        recs[r["name"]] = r
    ordered = [recs[n] for n in ORDER if n in recs]
    for n, r in recs.items():
        if n not in ORDER:
            ordered.append(r)
    return ordered


def bar(ax, recs, getter, title, ylabel, hline=None, label_fn=lambda v: f"{v:.2f}"):
    names = [LABELS.get(r["name"], r["name"]) for r in recs]
    vals = [getter(r) for r in recs]
    colors = [COLORS.get(r["name"], "#666") for r in recs]
    xs = np.arange(len(names))
    bars = ax.bar(xs, [v if v is not None else 0 for v in vals], color=colors)
    for x, v, b in zip(xs, vals, bars):
        if v is None:
            ax.text(x, 0.02 * (max([vv for vv in vals if vv is not None] or [1])), "OOM / N/A",
                    ha="center", va="bottom", fontsize=10, color="#aa0000", fontweight="bold")
        else:
            ax.text(x, v, label_fn(v), ha="center", va="bottom", fontsize=10)
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1)
        ax.text(len(names) - 0.5, hline, f" {hline}GB target", color="red", va="bottom", ha="right", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def chart_vram(recs, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    bar(ax, recs,
        lambda r: r["vram"]["peak_during_gsm8k_gb"] if r["vram"] and r["vram"].get("peak_during_gsm8k_gb") else None,
        "Peak VRAM During Inference", "GB", hline=8.0,
        label_fn=lambda v: f"{v:.2f} GB")
    fig.tight_layout()
    fig.savefig(out / "vram.png", dpi=150)
    plt.close(fig)


def chart_latency(recs, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    bar(ax, recs,
        lambda r: r["latency"]["p50_per_token_ms"] if r["latency"] else None,
        "Per-Token Latency (greedy decode)", "ms / token (p50)",
        label_fn=lambda v: f"{v:.1f} ms")
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=150)
    plt.close(fig)


def chart_throughput(recs, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in recs:
        if not r.get("throughput"):
            continue
        bs = [t["batch_size"] for t in r["throughput"] if not t.get("oom")]
        tps = [t["tokens_per_s"] for t in r["throughput"] if not t.get("oom")]
        if bs:
            ax.plot(bs, tps, marker="o", label=LABELS.get(r["name"], r["name"]),
                    color=COLORS.get(r["name"], "#666"), linewidth=2)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput vs Batch Size")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=150)
    plt.close(fig)


def chart_accuracy(recs, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    bar(ax, recs,
        lambda r: r["accuracy"]["accuracy"] if r.get("accuracy") else None,
        "GSM8K Pass@1 Accuracy", "accuracy",
        label_fn=lambda v: f"{v*100:.1f}%")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=150)
    plt.close(fig)


def chart_energy(recs, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    bar(ax, recs,
        lambda r: r.get("joules_per_correct"),
        "Energy per Correct Answer", "joules / correct",
        label_fn=lambda v: f"{v:.0f} J")
    fig.tight_layout()
    fig.savefig(out / "energy.png", dpi=150)
    plt.close(fig)


def chart_difficulty(recs, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    bucket_order = ["easy (≤2)", "med (3-4)", "hard (5-6)", "v.hard (7+)"]
    for r in recs:
        acc = r.get("accuracy") or {}
        bd = acc.get("by_difficulty")
        if not bd:
            continue
        ys = [bd[b]["accuracy"] if b in bd else None for b in bucket_order]
        xs = [b for b, y in zip(bucket_order, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, marker="o", label=LABELS.get(r["name"], r["name"]),
                color=COLORS.get(r["name"], "#666"), linewidth=2)
    ax.set_xlabel("Problem Difficulty (# reasoning steps)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Problem Difficulty (degradation curve)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_difficulty.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="bench_results")
    p.add_argument("--output", default="bench_results/charts")
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also pull teacher.json from repo root if it exists
    recs = load_records(in_dir)
    teacher_path = Path("teacher.json")
    if teacher_path.exists() and not any(r["name"] == "teacher" for r in recs):
        with open(teacher_path) as f:
            recs.insert(0, json.load(f))

    if not recs:
        print(f"No JSONs found in {in_dir}")
        return

    chart_vram(recs, out_dir)
    chart_latency(recs, out_dir)
    chart_throughput(recs, out_dir)
    chart_accuracy(recs, out_dir)
    chart_energy(recs, out_dir)
    chart_difficulty(recs, out_dir)
    print(f"Wrote 6 charts to {out_dir}")


if __name__ == "__main__":
    main()
