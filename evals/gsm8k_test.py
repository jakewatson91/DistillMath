import torch
from torch.utils.data import Dataset, DataLoader
from datasets import concatenate_datasets
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class GSM8K_Test:
    def __init__(self):
        self.test_dataset = load_dataset("openai/gsm8k", 'main')['test']
        self.prompt_head = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        self.prompt_tail = "<|im_end|>\n<|im_start|>assistant\n"
        self.questions = [self.prompt_head + ex["question"] + self.prompt_tail for ex in self.test_dataset]
        self.answers = [re.search(r"####\s*([^\s]+)", ex["answer"]) for ex in self.test_dataset]
        self.answers = [match.group(1) for match in self.answers]
        assert self.answers.count(None) == 0, "Some answers are not in the expected format."
        assert len(self.questions) == len(self.answers) == len(self.test_dataset), "Length mismatch among questions, answers and dataset."
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")

    def test_pass1(self, model, batch_size, device):
        model.eval() # hugging face model
        self.tokenizer.padding_side = "left"
        correct = 0
        for i in tqdm(range(0, len(self.test_dataset), batch_size)):
            questions, answers = self.questions[i:i+batch_size], self.answers[i:i+batch_size]
            model_inputs = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_k=32,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for output, answer in zip(outputs, answers):
                match = re.findall(r"\\boxed\{([^}]*)\}", output)
                if match:
                    pred = match[-1].strip()
                    if pred == answer:
                        correct += 1
        return correct / len(self.test_dataset)
    
if __name__ == "__main__":
    from transformers import AutoTokenizer

    gsm8k_test = GSM8K_Test()
    teacher_model = "Qwen/Qwen2.5-Math-7B-Instruct"
    tokenizer = "Qwen/Qwen2.5-Math-7B-Instruct"
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    accuracy = gsm8k_test.test_pass1(model=model, batch_size=128, device=device)
    print(f"Accuracy: {accuracy}")