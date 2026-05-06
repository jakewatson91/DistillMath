import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import concatenate_datasets
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
import re
import json

HF_MODEL_REPO = "jakewatson91/mathdistill-model"


def _hf_login():
    load_dotenv()
    token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

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
        collect_res_correct, collect_res_wrong = [], []
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
                        collect_res_correct.append({'question': output, 'correct answer': answer})
                    else:
                        collect_res_wrong.append({'question': output, 'correct answer': answer})
        with open("test_" + "_math_correct.json", "w") as f:
            json.dump(collect_res_correct, f, indent=4)
        with open("test_" + "_math_wrong.json", "w") as f:
            json.dump(collect_res_wrong, f, indent=4)
        return correct / len(self.test_dataset)
    
if __name__ == "__main__":
    from transformers import AutoTokenizer

    _hf_login()
    gsm8k_test = GSM8K_Test()
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_REPO,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)

    accuracy = gsm8k_test.test_pass1(model=model, batch_size=128, device=device)
    print(f"Accuracy: {accuracy}")