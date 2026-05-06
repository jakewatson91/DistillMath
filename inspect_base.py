"""Sample a few generations from the base student model to verify output format.

Usage:
    python inspect_base.py
    python inspect_base.py --model jakewatson91/mathdistill-model  # for comparison
    python inspect_base.py --n 10
"""

import argparse
import os
import random
import re

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--random", action="store_true", help="sample from random indices instead of the first N")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    load_dotenv()
    token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    ds = load_dataset("openai/gsm8k", "main")["test"]

    if args.random:
        random.seed(args.seed)
        indices = random.sample(range(len(ds)), args.n)
    else:
        indices = list(range(args.n))

    n_with_box = 0
    n_correct = 0
    wrong_examples = []

    for i, idx in enumerate(indices):
        ex = ds[idx]
        question = ex["question"]
        gold = re.search(r"####\s*([^\s]+)", ex["answer"]).group(1).strip().replace(",", "")

        prompt = PROMPT_HEAD + question + PROMPT_TAIL
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_only = out[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(gen_only, skip_special_tokens=True)
        boxed = re.findall(r"\\boxed\{([^}]*)\}", decoded)
        pred = boxed[-1].strip().replace(",", "") if boxed else None
        is_correct = pred == gold

        if boxed:
            n_with_box += 1
        if is_correct:
            n_correct += 1
        else:
            wrong_examples.append((idx, gold, pred, decoded[-200:]))

        print("\n" + "=" * 80)
        print(f"Q{i+1} (idx={idx}): {question}")
        print(f"GOLD: {gold}")
        print("-" * 80)
        print("MODEL OUTPUT:")
        print(decoded)
        print("-" * 80)
        print(f"  \\boxed{{}} count: {len(boxed)}")
        print(f"  predicted: {pred}")
        print(f"  correct: {is_correct}")

    print("\n" + "=" * 80)
    print(f"SUMMARY ({args.n} samples, indices={'random' if args.random else 'first'}):")
    print(f"  with \\boxed{{}}: {n_with_box}/{args.n} ({100*n_with_box/args.n:.0f}%)")
    print(f"  correct:        {n_correct}/{args.n} ({100*n_correct/args.n:.0f}%)")

    if wrong_examples:
        print("\nWRONG ANSWERS — gold vs pred (last 200 chars of output):")
        for idx, gold, pred, tail in wrong_examples:
            print(f"  idx={idx}: gold={gold!r}, pred={pred!r}")
            print(f"    ...{tail}")


if __name__ == "__main__":
    main()
