"""Side-by-side comparison: distilled vs Qwen 1.5B Instruct on a few GSM8K questions.

Picks N questions across difficulty buckets, runs both models with greedy decode,
writes a markdown file you can paste into slides.

Usage:
    python side_by_side.py
    python side_by_side.py --n 5 --output side_by_side.md
    python side_by_side.py --indices 0,42,123,456,789  # specific GSM8K indices
"""

import argparse
import gc
import json
import os
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_HEAD = (
    "<|im_start|>system\nPlease reason step by step, "
    "and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"

MODELS = [
    ("Distilled 1.5B", "jakewatson91/mathdistill-model"),
    ("Qwen 1.5B Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
]


def hf_login():
    load_dotenv()
    token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)


def difficulty_label(answer: str) -> str:
    steps = len(re.findall(r"<<.*?>>", answer))
    if steps <= 2:
        return f"easy ({steps} steps)"
    if steps <= 4:
        return f"medium ({steps} steps)"
    if steps <= 6:
        return f"hard ({steps} steps)"
    return f"very hard ({steps} steps)"


def pick_diverse_indices(ds, n, seed):
    """Pick n indices spanning easy → very hard."""
    by_bucket = {"easy": [], "med": [], "hard": [], "vhard": []}
    for i, ex in enumerate(ds):
        s = len(re.findall(r"<<.*?>>", ex["answer"]))
        if s <= 2:
            by_bucket["easy"].append(i)
        elif s <= 4:
            by_bucket["med"].append(i)
        elif s <= 6:
            by_bucket["hard"].append(i)
        else:
            by_bucket["vhard"].append(i)

    rng = random.Random(seed)
    picks = []
    # Distribute across buckets, weighted toward harder ones for visual contrast
    weights = {"easy": 1, "med": 2, "hard": 1, "vhard": 1}
    plan = []
    for bucket, count in weights.items():
        plan.extend([bucket] * count)
    while len(plan) < n:
        plan.append("med")
    plan = plan[:n]

    for bucket in plan:
        pool = by_bucket[bucket]
        if not pool:
            pool = by_bucket["med"] or by_bucket["easy"]
        idx = rng.choice(pool)
        picks.append(idx)
        pool.remove(idx)
    return sorted(picks)


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen_only = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True)


def extract_answer(text):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return None
    return matches[-1].strip().replace(",", "")


def run_model(model_id, prompts, max_new_tokens, device):
    print(f"\n=== Loading {model_id} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    outputs = []
    for i, p in enumerate(prompts):
        print(f"  generating {i+1}/{len(prompts)}...")
        outputs.append(generate(model, tokenizer, p, max_new_tokens, device))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--indices", default="", help="comma-separated GSM8K test indices to override auto-pick")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="side_by_side.md")
    p.add_argument("--json-output", default="side_by_side.json")
    args = p.parse_args()

    hf_login()
    ds = load_dataset("openai/gsm8k", "main")["test"]

    if args.indices:
        indices = [int(x) for x in args.indices.split(",")]
    else:
        indices = pick_diverse_indices(ds, args.n, args.seed)
    print(f"Selected indices: {indices}")

    questions, golds, prompts, diffs = [], [], [], []
    for idx in indices:
        ex = ds[idx]
        questions.append(ex["question"])
        m = re.search(r"####\s*([^\s]+)", ex["answer"])
        golds.append(m.group(1).strip().replace(",", ""))
        prompts.append(PROMPT_HEAD + ex["question"] + PROMPT_TAIL)
        diffs.append(difficulty_label(ex["answer"]))

    # Run each model in turn
    results = {}
    for label, model_id in MODELS:
        results[label] = run_model(model_id, prompts, args.max_new_tokens, args.device)

    # Compose markdown
    lines = ["# Side-by-side: Distilled vs Qwen Instruct\n"]
    json_dump = []

    for q_i, idx in enumerate(indices):
        q = questions[q_i]
        gold = golds[q_i]
        diff = diffs[q_i]

        lines.append(f"\n---\n\n## Q{q_i+1} — {diff} (gsm8k idx {idx})\n")
        lines.append(f"**Question:** {q}\n")
        lines.append(f"**Gold answer:** `{gold}`\n")
        lines.append("| Model | Predicted | Correct |")
        lines.append("|---|---|---|")

        record = {
            "index": idx,
            "difficulty": diff,
            "question": q,
            "gold": gold,
            "responses": {},
        }

        for label, _ in MODELS:
            resp = results[label][q_i]
            pred = extract_answer(resp)
            correct = pred == gold
            mark = "✅" if correct else "❌"
            lines.append(f"| {label} | `{pred}` | {mark} |")
            record["responses"][label] = {
                "full": resp, "predicted": pred, "correct": correct
            }
        lines.append("")

        for label, _ in MODELS:
            resp = results[label][q_i]
            lines.append(f"**{label} response:**\n")
            lines.append("```")
            lines.append(resp.strip())
            lines.append("```\n")

        json_dump.append(record)

    Path(args.output).write_text("\n".join(lines))
    Path(args.json_output).write_text(json.dumps(json_dump, indent=2))
    print(f"\n✓ Wrote {args.output} and {args.json_output}")

    # Print compact summary to stdout
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for q_i, idx in enumerate(indices):
        print(f"\nQ{q_i+1} idx={idx} ({diffs[q_i]}) gold={golds[q_i]}")
        for label, _ in MODELS:
            resp = results[label][q_i]
            pred = extract_answer(resp)
            mark = "✓" if pred == golds[q_i] else "✗"
            print(f"  {mark} {label:25s} → {pred}")


if __name__ == "__main__":
    main()
