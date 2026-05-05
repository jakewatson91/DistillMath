from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm

dataset_name = "meta-math/MetaMathQA"
if dataset_name == "openai/gsm8k":
    question_field = "question"
    n = 4 # number of generations per prompt
elif dataset_name == "meta-math/MetaMathQA":
    question_field = "original_question"
    n = 2
else:
    raise NotImplementedError(f"Dataset {dataset_name} not supported yet.")
####### Model Configuration ########
teacher_model = "Qwen/Qwen2.5-Math-7B-Instruct"
tokenizer = "Qwen/Qwen2.5-Math-7B-Instruct"
tensor_parallel_size = 2
dtype = "bfloat16"
enable_prefix_caching = False
trust_remote_code = False
####################################
max_tokens = 2000
collect_temperature = 0.7
top_p = 0.8
##################################
gen_batch_size = 1
#################################
teacher_llm = LLM(model=teacher_model,
                  tokenizer=tokenizer,
                  tensor_parallel_size=tensor_parallel_size,
                  dtype=dtype,
                  enforce_eager=True,
                  disable_custom_all_reduce=True,
                  enable_prefix_caching=enable_prefix_caching,
                  trust_remote_code=trust_remote_code,)
sampling_params = SamplingParams(n=n,
                                max_tokens=max_tokens,
                                temperature=collect_temperature,
                                top_p=top_p,
                                frequency_penalty=0,
                                presence_penalty=0,
                                stop=['###'],)

# Load the dataset
if dataset_name == "openai/gsm8k":
    dataset = load_dataset("openai/gsm8k", 'main')['train']
elif dataset_name == "meta-math/MetaMathQA":
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
else:
    raise NotImplementedError(f"Dataset {dataset_name} not supported yet.")
prompt_head = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
prompt_tail = "<|im_end|>\n<|im_start|>assistant\n"

exec_batch = []
collect_res = []
for instance in tqdm(dataset):
    # question = instance["instruction"]
    question = instance[question_field]
    prompt = prompt_head + question + prompt_tail
    exec_batch.append(prompt)
outputs = teacher_llm.generate(exec_batch, sampling_params)

for prompt, output in zip(exec_batch, outputs):
    res = {'prompt': prompt, 'response': []}
    for o in output.outputs:
        res['response'].append(o.text)
    collect_res.append(res)

with open("collect_" + dataset_name.replace("/", "_") + "_math.json", "w") as f:
    json.dump(collect_res, f, indent=4)