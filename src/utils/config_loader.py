import yaml
from pydantic import BaseModel, Field
from typing import Optional

class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0

class QuantConfig(BaseModel):
    use_4bit: bool = True

class StudentModelConfig(BaseModel):
    name: str
    max_seq_length: int = 512
    quant: QuantConfig = Field(default_factory=QuantConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)

class TeacherModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

class ModelsConfig(BaseModel):
    student: StudentModelConfig
    teacher: TeacherModelConfig = Field(default_factory=TeacherModelConfig)

class TrainingConfig(BaseModel):
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    max_steps: int = 500
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    logging_steps: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 3407
    report_to: str = "none"
    loss_alpha: float = 0.5
    temperature: float = 2.0

class Config(BaseModel):
    models: ModelsConfig
    training: TrainingConfig

def load_model_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)