import torch
from pipp.pipp_utils import pipp_setup, pipp_cleanup
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe
import yaml
import math
import deepspeed
from deepspeed.pipe import PipelineModule
import torch.nn as nn

def pipp_main(rank: int, world_size: int, resume: bool):
    # configure
    # with open("pretrain_config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    # Grab the model on meta device
    # config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    # with torch.device("meta"):
    #     teacher_model = AutoModelForCausalLM.from_config(config)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct", low_cpu_mem_usage=True
    )
    teacher_model.config.use_cache = True  # KV-cache logic breaks graphs
    teacher_model.config.return_dict = True # Dicts are hard to trace
    # teacher_model.config._attn_implementation = "sdpa"
    # print(teacher_model)
    # teacher_model = PipelineWrapper(teacher_model)
    # Cut model by equal number of layers per rank
    layers = [teacher_model.model.embed_tokens] + list(teacher_model.model.layers) + [teacher_model.model.norm, teacher_model.lm_head]
    # --- Wrap in PipelineModule ---
    pipe_model = PipelineModule(
        layers=layers,
        loss_fn=nn.CrossEntropyLoss(),
        num_stages=3,
        partition_method="parameters"
    )

    # --- Initialize DeepSpeed ---
    engine, optimizer, _, _ = deepspeed.initialize(
        model=pipe_model,
        model_parameters=[p for p in pipe_model.parameters() if p.requires_grad],
        config="ds_config.json"
    )
    engine.eval()  # we only do inference in pipp_main

    print("✅ DeepSpeed pipeline test successful")

    # --- Example prompts ---
    prompts = [
        "Write a Python function to compute factorial",
        "Explain pipeline parallelism in deep learning"
    ]

    # Tokenize prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(f'cuda:{rank}')
    labels = input_ids.clone()  # LM next-token prediction

    # --- Create microbatches ---
    microbatch_size = 1  # adjust for memory
    microbatches_list = []
    for i in range(0, input_ids.size(0), microbatch_size):
        microbatches_list.append({
            'inputs': input_ids[i:i+microbatch_size],
            'labels': labels[i:i+microbatch_size]
        })

    # Convert list to iterator
    microbatches = iter(microbatches_list)

    # --- Run train_batch ---
    loss = engine.train_batch(microbatches)
    print(f"[Rank {rank}] Train batch loss: {loss.item()}")





    # layers_per_rank = teacher_model.config.num_hidden_layers // world_size
    # print(f"layers_per_rank = {layers_per_rank}")
    # # Create a pipeline representation from the model
    # mb_prompts = (
    #     "How do you", "I like to",
    # )  # microbatch size = 2
    # mb_inputs = tokenizer(mb_prompts, return_tensors="pt", padding=True)
    # split_spec = {
    #     f"model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
    #     for i in range(1, world_size)
    # }
    # pipe = pipeline(teacher_model, mb_args=(mb_inputs["input_ids"],), split_spec=split_spec)
    # # Create pipeline stage for each rank
    # stage = pipe.build_stage(rank, device=f'cuda:{rank}')

    pipp_cleanup()