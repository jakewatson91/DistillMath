import os
from datasets import load_dataset
import torch
from transformers import Trainer, TrainingArguments, DefaultDataCollator
import torch.nn.functional as F
from typing import Union, List, Optional, Callable, Any

class TeacherDistiller:
    def __init__(self, teacher_model, tokenizer, max_length=512):
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        texts = [f["text"] for f in features]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)

        teacher_inputs = {k: v.to(self.teacher_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = outputs.logits
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "teacher_logits": teacher_logits.cpu()
        }
    
class InstilTrainer(Trainer):
    def __init__(self, *args, loss_alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_alpha = loss_alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_logits = inputs.pop("teacher_logits")

        outputs = model(**inputs)
        student_logits = outputs.logits

        teacher_logits = teacher_logits.to(student_logits.device)

        T = self.temperature
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)

        sft_loss = outputs.loss
        loss = (self.loss_alpha * distill_loss) + ((1 - self.loss_alpha) * sft_loss)
        
        return (loss, outputs) if return_outputs else loss

def train(
    *,
    student_model,
    teacher_model,
    tokenizer,
    train_dataset,
    output_dir: str,
    loss_alpha: float = 0.5,
    temperature: float = 2.0,
    eval_dataset = None,
    **kwargs 
):
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=output_dir,
        **kwargs
    )

    trainer = InstilTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_alpha=loss_alpha,    
        temperature=temperature,
        data_collator=TeacherDistiller(teacher_model, tokenizer),
    )

    trainer.train()
    return trainer

def save_lora_adapter(model, tokenizer, adapter_dir: str):
    # Always save tokenizer with adapter for reproducibility
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

def merge_and_save_hf(model, tokenizer, merged_dir: str, method: str = "merged_16bit"):
    print(f"Merging model using method: {method}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method=method)