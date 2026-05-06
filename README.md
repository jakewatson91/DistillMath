# MathDistill

Knowledge distillation of `Qwen/Qwen2.5-Math-7B-Instruct` (teacher, 7B) into `Qwen/Qwen2.5-Math-1.5B` (student, 1.5B) on GSM8K + MetaMathQA, targeting deployment on consumer-grade hardware (≤8GB VRAM) where the teacher cannot fit.

The repo has two pipelines and a benchmarking layer:
- **collect** — teacher generates training traces (top-k logits + sampled completions) on the math corpora
- **distill** — student trains on those traces with top-k KL + cross-entropy loss
- **benchmark** — distilled student vs base student vs teacher on latency, throughput, peak VRAM, energy per correct answer, and GSM8K accuracy bucketed by problem difficulty

## Results

Measured on a single NVIDIA A30 with VRAM capped at 8 GB to simulate consumer-grade hardware. GSM8K test set (1319 questions), greedy decode, bf16.

| Metric | Teacher 7B | Qwen 1.5B Instruct | **Distilled 1.5B** |
|---|---|---|---|
| GSM8K pass@1 (greedy) | 85.4% (cited) | 66.3% (874/1319) | **80.5% (1062/1319)** |
| Peak VRAM | OOM (>8 GB) | 3.31 GB | **3.29 GB** |
| Latency p50 | — | 17.17 ms/tok | **17.09 ms/tok** |
| Latency p95 | — | 17.21 ms/tok | **17.13 ms/tok** |
| Throughput @ batch=1 | — | 57.8 tok/s | **58.5 tok/s** |
| Throughput @ batch=8 | — | 393.6 tok/s | **395.4 tok/s** |
| Throughput @ batch=16 | — | 789.7 tok/s | **794.0 tok/s** |
| Avg power draw | — | 114.3 W | **113.4 W** |
| Energy / correct answer | — | 196.6 J | **158.1 J** |

Headline takeaways:
- **Teacher cannot run on the 8 GB target.** The distilled student is the only feasible option under this hardware cap.
- **Distillation lifts GSM8K accuracy by +14.3 pp** over an off-the-shelf Qwen 1.5B Instruct (66.3% → 80.5%) while preserving an essentially identical compute profile (same architecture, weights only differ).
- **Energy per correct answer drops 20%** (196.6 J → 158.1 J). Both models burn similar total energy on the eval; distilled produces more useful output per joule.

Accuracy by problem difficulty (gap widens as problems get harder — distillation transfers most on the easy/med buckets):

| Difficulty (reasoning steps) | Qwen 1.5B Instruct | Distilled 1.5B | Δ |
|---|---|---|---|
| Easy (≤2) | 73.4% (323/440) | 87.7% (386/440) | +14.3 |
| Medium (3–4) | 67.6% (442/654) | 80.3% (525/654) | +12.7 |
| Hard (5–6) | 49.2% (96/195) | 68.7% (134/195) | +19.5 |
| Very hard (7+) | 43.3% (13/30) | 56.7% (17/30) | +13.4 |

## Architecture

```
              GSM8K + MetaMathQA
                      │
                      ▼
         ┌────────────────────────────┐
         │  Teacher: Qwen2.5-Math-7B  │   generates top-k logits +
         │           -Instruct        │   sampled chain-of-thought
         │           (frozen, eval)   │   completions per question
         └─────────────┬──────────────┘
                       │
                       ▼
         ┌────────────────────────────┐
         │  Teacher traces (JSON)     │  ──►  jakewatson91/mathdistill-traces
         └─────────────┬──────────────┘
                       │
   Student init ──►    ▼
   Qwen2.5-Math-1.5B
   (base)    ┌────────────────────────────┐
             │  Distillation              │   • top-k KL + cross-entropy
             │                            │   • temperature 2.0, top-k 32
             │                            │   • bf16, multi-GPU TP
             └─────────────┬──────────────┘
                           │
                           ▼
             ┌────────────────────────────┐
             │  Distilled student (1.5B)  │  ──►  jakewatson91/mathdistill-model
             │  Fits in <8 GB VRAM        │
             └────────────────────────────┘
```

The teacher group runs tensor-parallel across multiple GPUs and emits top-k logit distributions per token; the student group consumes those distributions to compute KL loss against its own predictions, plus cross-entropy against the sampled token. Configuration in `distill_config.yaml`.

## Hosted artifacts

- **Distilled model + tokenizer**: [`jakewatson91/mathdistill-model`](https://huggingface.co/jakewatson91/mathdistill-model)
- **Teacher traces (training data)**: [`jakewatson91/mathdistill-traces`](https://huggingface.co/datasets/jakewatson91/mathdistill-traces)

The training and eval scripts pull from these directly. Auth is read from a `.env` at the repo root:

```
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

## Training

```bash
python distill.py
# resume from checkpoint
python distill.py --resume
```

Loads the teacher sharded across the GPUs in `teacher_group`, the student on the GPUs in `student_group`, pulls training JSONs from `jakewatson91/mathdistill-traces`, and distills via top-k KL + CE. Multi-GPU is required: teacher needs ~14GB and runs in parallel with the student. Edit `distill_config.yaml` for GPU groups, batch size, learning rate, KL weight, and temperature.

## Benchmarking

`benchmark.py` enforces an 8GB VRAM cap by default via `torch.cuda.set_per_process_memory_fraction`. Allocations beyond 8GB raise a real CUDA OOM regardless of the underlying GPU's true size, so results reflect the deployment constraint rather than the test hardware.

```bash
python benchmark.py --model Qwen/Qwen2.5-Math-1.5B           --name base_student
python benchmark.py --model jakewatson91/mathdistill-model     --name distilled
python benchmark.py --model Qwen/Qwen2.5-Math-7B-Instruct    --name teacher
```

Each run writes one JSON to `bench_results/`. The teacher run OOMs at model load under the 8GB cap — captured as `vram.oom_on_8gb: true` in `teacher.json`, with accuracy cited from prior measurement (85.4% on GSM8K).

Smoke test:
```bash
python benchmark.py --model Qwen/Qwen2.5-Math-1.5B --name base_student --gsm8k-limit 50
```

Flags:

| Flag | Default | Effect |
|---|---|---|
| `--vram-cap-gb` | 8.0 | Cap process VRAM. Set 0 to disable. |
| `--gsm8k-limit` | 0 (full set) | Subsample for fast iteration |
| `--gsm8k-batch-size` | 8 | Eval batch size |
| `--batch-sizes` | `1,2,4,8,16` | Throughput sweep |
| `--n-latency-runs` | 30 | Latency sample count |
| `--skip-energy` | off | Skip pynvml power sampling |

### Charts

```bash
python plot_benchmarks.py
```

Reads `bench_results/*.json` plus `teacher.json` and writes to `bench_results/charts/`:

| File | Content |
|---|---|
| `headline.png` | Capability vs cost — VRAM × accuracy scatter with infeasible region shaded past 8 GB and an arrow showing the accuracy lift from distillation |
| `throughput.png` | Tokens/s vs batch size, log x-axis |
| `degradation.png` | Accuracy by problem difficulty — Qwen 1.5B Instruct vs distilled 1.5B across reasoning-step buckets |
| `summary.md` | Full metric table (latency, throughput, VRAM, energy, accuracy) across all three models |

### Methodology

- **Decode**: greedy (`do_sample=False`) across all measurements to remove sampling variance.
- **Latency**: 30 runs of fixed-length generation after 5 warmup runs, p50/p95 reported per generated token.
- **Throughput**: sweep over batch sizes; auto-stops and records the OOM batch size.
- **Energy**: GPU power sampled at 10Hz via `pynvml.nvmlDeviceGetPowerUsage` over the GSM8K eval run, integrated trapezoidally to joules, then divided by # correct → joules per useful answer.
- **Difficulty buckets** for the degradation chart are based on the count of `<<...>>` calculator tags in each GSM8K gold answer (≤2 easy, 3–4 med, 5–6 hard, 7+ very hard). Step count is a coarse proxy for problem complexity, not a canonical difficulty label.
- **Teacher accuracy** is cited (85.4%); the 7B teacher does not load under an 8GB cap.

## Repo map

All scripts assume they're run from the repo root (so `.env` and relative paths resolve correctly).

```
distill.py                       training entrypoint (multi-GPU spawn)
distill_config.yaml              training config (GPU groups, batch size, LR, KL weight, temperature)
benchmark.py                     single-model benchmark: latency, throughput, VRAM, energy, accuracy
plot_benchmarks.py               benchmark JSONs → 3 charts + summary.md
teacher.json                     teacher record (OOM markers + cited accuracy)
requirements.txt                 non-torch deps
.env                             HF_API_KEY=...  (gitignored)

dist_distill/                    distillation core (trainer, distiller, distributed setup, TP plan)
dataset/data_loader.py           DistillDataset over the collected JSON traces
collect.py / collect_math.py     collect-side scripts (teacher trace generation)
pipp/                            pipeline-parallel utilities used by training

evals/                           legacy single-model eval scripts (kept for reference)
  gsm8k_test.py                  GSM8K eval — teacher (older, pre-benchmark.py)
  gsm8k_test_res.py              GSM8K eval — distilled student (pulls from HF)
  code_test.py                   LiveCodeBench eval (separate code-generation pipeline)
  test.py                        scratch model-loading test

tools/                           diagnostic + comparison utilities
  inspect_base.py                sample N outputs from a model, count format compliance
  side_by_side.py                distilled vs Qwen Instruct on N GSM8K questions → markdown

bench_results/                   benchmark JSONs + generated charts (tracked — these are the deliverable)
  base_student.json
  distilled.json
  charts/
    headline.png
    throughput.png
    degradation.png
    summary.md
```
