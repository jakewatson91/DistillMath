import os
import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_student_model(model_config: Any, inference_mode: bool = True) -> Tuple[Any, Any]:
    """
    Loads a model using Unsloth.
    Args:
        model_config: ModelConfig Pydantic object
        inference_mode: If True, optimizes for inference (faster). If False, prepares for training (LoRA).
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model.name,
        max_seq_length=model_config.model.max_seq_length,
        dtype=None, # Auto-detect
        load_in_4bit=model_config.quant.use_4bit,
        device_map={"": "cuda:1"}
    )

    if inference_mode:
        FastLanguageModel.for_inference(model)
    else:
        lora = model_config.lora
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora.r if lora else 16,
            lora_alpha=lora.alpha if lora else 16,
            lora_dropout=lora.dropout if lora else 0,
            bias="none",
        )
    
    return model, tokenizer

def load_teacher_model(model_name: str):
    # --- Load model across multiple GPUs ---
    device_map = {
        "model.embed_tokens": 0,
    }
    for i in range(14): device_map[f"model.layers.{i}"] = 0
    for i in range(14, 28): device_map[f"model.layers.{i}"] = 1
    device_map["model.norm"] = 1
    device_map["lm_head"] = 1
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,        # manually specify device map
        torch_dtype=torch.float16, # use FP16 to save memory
        low_cpu_mem_usage=True     # avoids huge RAM usage
    )
    model.eval()  # inference mode