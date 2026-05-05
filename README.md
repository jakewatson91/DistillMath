# MathDistill

Knowledge distillation of `Qwen/Qwen2.5-Math-7B-Instruct` (teacher, 7B) into `Qwen/Qwen2.5-Math-1.5B` (student, 1.5B) on GSM8K + MetaMathQA, targeting deployment on consumer-grade hardware (≤8GB VRAM) where the teacher cannot fit.

The repo has two pipelines and a benchmarking layer:
- **collect** — teacher generates training traces (top-k logits + sampled completions) on the math corpora
- **distill** — student trains on those traces with top-k KL + cross-entropy loss
- **benchmark** — distilled student vs base student vs teacher on latency, throughput, peak VRAM, energy per correct answer, and GSM8K accuracy bucketed by problem difficulty

## Hosted artifacts

- **Distilled model + tokenizer**: [`jakewatson/mathdistill-model`](https://huggingface.co/jakewatson/mathdistill-model)
- **Teacher traces (training data)**: [`jakewatson/mathdistill-traces`](https://huggingface.co/datasets/jakewatson/mathdistill-traces)

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

Loads the teacher sharded across the GPUs in `teacher_group`, the student on the GPUs in `student_group`, pulls training JSONs from `jakewatson/mathdistill-traces`, and distills via top-k KL + CE. Multi-GPU is required: teacher needs ~14GB and runs in parallel with the student. Edit `distill_config.yaml` for GPU groups, batch size, learning rate, KL weight, and temperature.

## Benchmarking

`benchmark.py` enforces an 8GB VRAM cap by default via `torch.cuda.set_per_process_memory_fraction`. Allocations beyond 8GB raise a real CUDA OOM regardless of the underlying GPU's true size, so results reflect the deployment constraint rather than the test hardware.

```bash
python benchmark.py --model Qwen/Qwen2.5-Math-1.5B           --name base_student
python benchmark.py --model jakewatson/mathdistill-model     --name distilled
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
| `summary.md` | Full metric table (latency, throughput, VRAM, energy, accuracy) across all three models |

### Methodology

- **Decode**: greedy (`do_sample=False`) across all measurements to remove sampling variance.
- **Latency**: 30 runs of fixed-length generation after 5 warmup runs, p50/p95 reported per generated token.
- **Throughput**: sweep over batch sizes; auto-stops and records the OOM batch size.
- **Energy**: GPU power sampled at 10Hz via `pynvml.nvmlDeviceGetPowerUsage` over the GSM8K eval run, integrated trapezoidally to joules, then divided by # correct → joules per useful answer.
- **Teacher accuracy** is cited (85.4%); the 7B teacher does not load under an 8GB cap.

## Repo map

```
distill.py                  training entrypoint (multi-GPU spawn)
distill_config.yaml         training config (GPU groups, batch size, LR, KL weight, temperature)
collect.py / collect_math.py   collect-side scripts (teacher trace generation)
dist_distill/               distillation core (trainer, distiller, distributed setup, TP plan)
dataset/data_loader.py      DistillDataset over the collected JSON traces
gsm8k_test.py               GSM8K eval — teacher
gsm8k_test_res.py           GSM8K eval — distilled student (pulls from HF)
code_test.py                LiveCodeBench eval (separate code-generation pipeline)
benchmark.py                single-model benchmark: latency, throughput, VRAM, energy, accuracy
plot_benchmarks.py          benchmark JSONs → 6 PNGs
teacher.json                teacher record (OOM markers + cited accuracy)
requirements.txt            non-torch deps
.env                        HF_API_KEY=...  (gitignored)
```
