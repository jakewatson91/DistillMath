import argparse
from pathlib import Path
from datasets import load_dataset
from src.utils.config_loader import load_model_config
from src.models.model_loader import load_student_model, load_teacher_model
from src.training.trainer import train

PROJECT_ROOT = Path(__file__).resolve().parent.parent
results_dir = PROJECT_ROOT / "results"
checkpoint_dir = results_dir / "checkpoints"
final_dir = results_dir / "final_student_model"
    
checkpoint_dir.mkdir(parents=True, exist_ok=True)
final_dir.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    
    dataset = load_dataset("Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b", split="train", streaming=True)
    
    def format_text(x):
        return {"text": f"### Instruction:\n{x['input']}\n\n### Response:\n{x['output']}"}

    train_dataset = (
        dataset
        .filter(lambda x: x.get("domain") == "code")
        .map(format_text)
        .take(1000)
    )

    student_model, tokenizer = load_student_model(cfg.models.student, inference_mode=False)
    teacher_model = load_teacher_model() 
    train(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset, 
        output_dir=str(checkpoint_dir),
        **cfg.training.model_dump()
    )
    
    print("Saving Final Model...")
    student_model.save_pretrained_merged(str(final_dir / "merged"), tokenizer, save_method="merged_16bit")
    student_model.save_pretrained(str(final_dir / "adapter"))
    tokenizer.save_pretrained(str(final_dir / "adapter"))

if __name__ == "__main__":
    main()