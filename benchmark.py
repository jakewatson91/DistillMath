"""Benchmark a single model on the 8GB target hardware.

Run once per model. Produces one JSON in --output. Then plot_benchmarks.py
reads all JSONs in that dir and emits charts.

Example:
    python benchmark.py --model jakewatson/mathdistill-model --name distilled
    python benchmark.py --model Qwen/Qwen2.5-Math-1.5B --name base_student
"""

import argparse
import gc
import json
import os
import re
import statistics
import threading
import time
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_HEAD = (
    "<|im_start|>system\nPlease reason step by step, "
    "and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"


def hf_login():
    load_dotenv()
    token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)


class PowerSampler:
    """Background thread that samples GPU power draw via pynvml at fixed Hz."""

    def __init__(self, gpu_index=0, hz=10):
        import pynvml

        self.pynvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.interval = 1.0 / hz
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            t = time.time()
            mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.samples.append((t, mw / 1000.0))  # watts
            time.sleep(self.interval)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        if len(self.samples) < 2:
            return {"energy_joules": 0.0, "avg_power_w": 0.0, "duration_s": 0.0}
        energy = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples[:-1], self.samples[1:]):
            energy += 0.5 * (w0 + w1) * (t1 - t0)  # trapezoidal
        duration = self.samples[-1][0] - self.samples[0][0]
        avg = energy / duration if duration > 0 else 0.0
        return {
            "energy_joules": energy,
            "avg_power_w": avg,
            "duration_s": duration,
            "n_samples": len(self.samples),
        }


def gsm8k_difficulty(answer: str) -> int:
    """Number of arithmetic steps in the gold answer (count of <<...>> tags)."""
    return len(re.findall(r"<<.*?>>", answer))


def difficulty_bucket(steps: int) -> str:
    if steps <= 2:
        return "easy (≤2)"
    if steps <= 4:
        return "med (3-4)"
    if steps <= 6:
        return "hard (5-6)"
    return "v.hard (7+)"


def load_gsm8k(prompt_head: str, prompt_tail: str):
    ds = load_dataset("openai/gsm8k", "main")["test"]
    questions, answers, steps = [], [], []
    for ex in ds:
        questions.append(prompt_head + ex["question"] + prompt_tail)
        m = re.search(r"####\s*([^\s]+)", ex["answer"])
        assert m
        answers.append(m.group(1).strip().replace(",", ""))
        steps.append(gsm8k_difficulty(ex["answer"]))
    return questions, answers, steps


@torch.inference_mode()
def measure_latency(model, tokenizer, prompt, device, n_runs=50, max_new_tokens=256, warmup=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    for _ in range(warmup):
        model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()

    times, gen_lens = [], []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        gen_len = out.shape[1] - inputs["input_ids"].shape[1]
        times.append(elapsed)
        gen_lens.append(gen_len)

    avg_gen = statistics.mean(gen_lens)
    return {
        "n_runs": n_runs,
        "max_new_tokens": max_new_tokens,
        "avg_gen_tokens": avg_gen,
        "p50_total_s": statistics.median(times),
        "p95_total_s": sorted(times)[int(0.95 * len(times)) - 1],
        "mean_total_s": statistics.mean(times),
        "p50_per_token_ms": 1000 * statistics.median(times) / avg_gen,
        "p95_per_token_ms": 1000 * sorted(times)[int(0.95 * len(times)) - 1] / avg_gen,
    }


@torch.inference_mode()
def measure_throughput(model, tokenizer, prompts, device, batch_sizes, max_new_tokens=256):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = []
    for bs in batch_sizes:
        batch = prompts[:bs]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            gen_tokens = (out.shape[1] - inputs["input_ids"].shape[1]) * bs
            results.append({
                "batch_size": bs,
                "elapsed_s": elapsed,
                "gen_tokens": gen_tokens,
                "tokens_per_s": gen_tokens / elapsed,
            })
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": bs, "oom": True})
            torch.cuda.empty_cache()
            break
    return results


@torch.inference_mode()
def measure_accuracy(model, tokenizer, questions, answers, steps, device, batch_size, max_new_tokens, sampler=None):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    by_bucket = {}  # bucket -> [n_correct, n_total]

    if sampler:
        sampler.start()

    for i in tqdm(range(0, len(questions), batch_size), desc="gsm8k"):
        qs = questions[i : i + batch_size]
        ans = answers[i : i + batch_size]
        st = steps[i : i + batch_size]
        inputs = tokenizer(qs, return_tensors="pt", padding=True, truncation=True).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for output, gold, s in zip(decoded, ans, st):
            bucket = difficulty_bucket(s)
            by_bucket.setdefault(bucket, [0, 0])
            by_bucket[bucket][1] += 1
            matches = re.findall(r"\\boxed\{([^}]*)\}", output)
            if matches:
                pred = matches[-1].strip().replace(",", "")
                if pred == gold:
                    correct += 1
                    by_bucket[bucket][0] += 1

    energy = sampler.stop() if sampler else None
    acc = correct / len(questions)
    bucket_acc = {b: {"correct": c, "total": t, "accuracy": c / t} for b, (c, t) in by_bucket.items()}

    return {
        "n_total": len(questions),
        "n_correct": correct,
        "accuracy": acc,
        "by_difficulty": bucket_acc,
        "energy": energy,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF repo or local path")
    p.add_argument("--name", required=True, help="label used in output filename")
    p.add_argument("--output", default="bench_results")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-sizes", default="1,2,4,8,16")
    p.add_argument("--n-latency-runs", type=int, default=30)
    p.add_argument("--latency-max-tokens", type=int, default=256)
    p.add_argument("--throughput-max-tokens", type=int, default=128)
    p.add_argument("--gsm8k-batch-size", type=int, default=8)
    p.add_argument("--gsm8k-max-tokens", type=int, default=512)
    p.add_argument("--gsm8k-limit", type=int, default=0, help="0 = full set")
    p.add_argument("--skip-energy", action="store_true")
    p.add_argument("--vram-cap-gb", type=float, default=8.0,
                   help="Cap process VRAM to simulate smaller GPU. Set 0 to disable.")
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    hf_login()

    if args.vram_cap_gb > 0:
        total_gb = torch.cuda.get_device_properties(args.device).total_memory / 1e9
        fraction = args.vram_cap_gb / total_gb
        torch.cuda.set_per_process_memory_fraction(fraction, device=args.device)
        print(f"VRAM capped at {args.vram_cap_gb} GB (fraction={fraction:.3f} of {total_gb:.1f} GB)")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.cuda.reset_peak_memory_stats(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    load_vram_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"VRAM after load: {load_vram_gb:.2f} GB")

    questions, answers, steps = load_gsm8k(PROMPT_HEAD, PROMPT_TAIL)
    if args.gsm8k_limit > 0:
        questions, answers, steps = questions[: args.gsm8k_limit], answers[: args.gsm8k_limit], steps[: args.gsm8k_limit]

    print("Measuring latency...")
    torch.cuda.reset_peak_memory_stats(args.device)
    latency = measure_latency(
        model, tokenizer, questions[0], args.device,
        n_runs=args.n_latency_runs, max_new_tokens=args.latency_max_tokens,
    )
    latency_peak_vram_gb = torch.cuda.max_memory_allocated(args.device) / 1e9

    print("Measuring throughput across batch sizes...")
    torch.cuda.reset_peak_memory_stats(args.device)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    throughput = measure_throughput(
        model, tokenizer, questions, args.device, batch_sizes, max_new_tokens=args.throughput_max_tokens
    )
    throughput_peak_vram_gb = torch.cuda.max_memory_allocated(args.device) / 1e9

    print("Measuring GSM8K accuracy + energy...")
    sampler = None if args.skip_energy else PowerSampler(gpu_index=int(args.device.split(":")[-1]))
    torch.cuda.reset_peak_memory_stats(args.device)
    accuracy = measure_accuracy(
        model, tokenizer, questions, answers, steps, args.device,
        batch_size=args.gsm8k_batch_size, max_new_tokens=args.gsm8k_max_tokens, sampler=sampler,
    )
    gsm8k_peak_vram_gb = torch.cuda.max_memory_allocated(args.device) / 1e9

    if accuracy["energy"] and accuracy["n_correct"] > 0:
        joules_per_correct = accuracy["energy"]["energy_joules"] / accuracy["n_correct"]
    else:
        joules_per_correct = None

    record = {
        "name": args.name,
        "model_id": args.model,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device),
        "gpu_total_vram_gb": torch.cuda.get_device_properties(args.device).total_memory / 1e9,
        "vram": {
            "after_load_gb": load_vram_gb,
            "peak_during_latency_gb": latency_peak_vram_gb,
            "peak_during_throughput_gb": throughput_peak_vram_gb,
            "peak_during_gsm8k_gb": gsm8k_peak_vram_gb,
        },
        "latency": latency,
        "throughput": throughput,
        "accuracy": accuracy,
        "joules_per_correct": joules_per_correct,
    }

    out_path = Path(args.output) / f"{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
