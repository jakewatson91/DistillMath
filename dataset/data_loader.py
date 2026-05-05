from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
import copy
import json
from transformers import AutoTokenizer

class DistillDataset(Dataset):
    def __init__(self, dataset_paths: list[str], tokenizer):
        self.tokenizer = tokenizer
        # print(self.tokenizer.all_special_tokens)
        # print(self.tokenizer.eos_token_id)
        self.dataset = []
        for dataset_path in dataset_paths:
            with open(dataset_path, "r") as f:
                data = json.load(f)
            self.dataset.extend(data)
        for sample in self.dataset:
            sample['response'] = list(set(sample['response']))  # remove duplicates
        idx_to_remove = []
        # for i, sample in enumerate(self.dataset):
        #     sample['response'] = list(filter(lambda s: s.endswith("```"), sample['response']))
        #     if len(sample['response']) == 0:
        #         idx_to_remove.append(i)
        # for idx in reversed(idx_to_remove):
        #     self.dataset.pop(idx)
        # dataset clean completed.
        total_question, total_response = 0, 0
        self.dataset_cleaned = []
        for sample in self.dataset:
            total_question += 1
            total_response += len(sample['response'])
            for response in sample['response']:
                self.dataset_cleaned.append({'prompt': sample['prompt'], 'response': response})
        assert len(self.dataset_cleaned) == total_response
        del self.dataset
        self.dataset = None
        print(f'There are {total_question} questions and {total_response} responses after cleaning.')

    def __len__(self):
        return len(self.dataset_cleaned)
    
    def __getitem__(self, idx):
        sample = self.dataset_cleaned[idx]
        question, response = sample['prompt'], sample['response'] + "<|endoftext|>"
        question_tok_np = np.array(self.tokenizer(question, truncation=False, max_length=None)['input_ids'])
        response_tok_np = np.array(self.tokenizer(response, truncation=False, max_length=None)['input_ids'])
        mask = np.ones(len(question_tok_np) + len(response_tok_np), dtype=np.int64)
        mask[0:len(question_tok_np)] = 0
        label = np.concatenate([question_tok_np, response_tok_np])
        label[0:len(question_tok_np)] = -100
        return np.concatenate([question_tok_np, response_tok_np]), mask, label
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    dataset = DistillDataset(["/home/zhiyang/projects/distill/dataset/collect_livecodebench_code_generation_lite.json", \
                              "/home/zhiyang/projects/distill/dataset/collect_bigcode_self-oss-instruct-sc2-exec-filter-50k.json", \
                              "/home/zhiyang/projects/distill/dataset/collect_HuggingFaceH4_CodeAlpaca_20K.json", \
                              "/home/zhiyang/projects/distill/dataset/collect_ise-uiuc_Magicoder-OSS-Instruct-75K.json"], tokenizer)
    # for sample in dataset:
    #     if sample['prompt'] == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\nYou will be given a question (problem specification) and will generate a correct program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\nQuestion: There is an N \\times N grid, where the cell at the i-th row from the top and the j-th column from the left contains the integer N \\times (i-1) + j.\nOver T turns, integers will be announced. On Turn i, the integer A_i is announced, and the cell containing A_i is marked. Determine the turn on which Bingo is achieved for the first time. If Bingo is not achieved within T turns, print -1.\nHere, achieving Bingo means satisfying at least one of the following conditions:\n\n- There exists a row in which all N cells are marked.\n- There exists a column in which all N cells are marked.\n- There exists a diagonal line (from top-left to bottom-right or from top-right to bottom-left) in which all N cells are marked.\n\nInput\n\nThe input is given from Standard Input in the following format:\nN T\r\nA_1 A_2 \\ldots A_T\n\nOutput\n\nIf Bingo is achieved within T turns, print the turn number on which Bingo is achieved for the first time; otherwise, print -1.\n\nConstraints\n\n\n- 2 \\leq N \\leq 2 \\times 10^3\n- 1 \\leq T \\leq \\min(N^2, 2 \\times 10^5)\n- 1 \\leq A_i \\leq N^2\n- A_i \\neq A_j if i \\neq j.\n- All input values are integers.\n\nSample Input 1\n\n3 5\r\n5 1 8 9 7\n\nSample Output 1\n\n4\r\n\nThe state of the grid changes as follows. Bingo is achieved for the first time on Turn 4.\n\nSample Input 2\n\n3 5\r\n4 2 9 7 5\n\nSample Output 2\n\n-1\r\n\nBingo is not achieved within five turns, so print -1.\n\nSample Input 3\n\n4 12\r\n13 9 6 5 2 7 16 14 8 3 10 11\n\nSample Output 3\n\n9\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n<|im_start|>assistant\n":
    #         print(sample['response'])
    #         break
    dataloader = DataLoader(dataset[:5], batch_size=32, shuffle=True)
    dataset_iter = iter(dataloader)
    for epoch in range(0, 4):
        print(epoch)
        while True:
            try:
                batch = next(dataset_iter)
                # process batch
                print(batch.shape)
            except StopIteration:
                print("-------------------")
                break