from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm

dataset_name = "ise-uiuc/Magicoder-OSS-Instruct-75K"
if dataset_name == "HuggingFaceH4/CodeAlpaca_20K":
    question_field = "prompt"
    n = 4 # number of generations per prompt
elif dataset_name == "bigcode/self-oss-instruct-sc2-exec-filter-50k":
    question_field = "instruction"
    n = 4
elif dataset_name == "livecodebench/code_generation_lite":
    question_field = "question_content"
    n = 4
elif dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
    question_field = "problem"
    n = 4
else:
    raise NotImplementedError(f"Dataset {dataset_name} not supported yet.")
####### Model Configuration ########
teacher_model = "Qwen/Qwen2.5-Coder-32B-Instruct"
tokenizer = "Qwen/Qwen2.5-Coder-32B-Instruct"
tensor_parallel_size = 2
dtype = "bfloat16"
enable_prefix_caching = False
trust_remote_code = False
####################################
max_tokens = 2000
collect_temperature = 0.2
top_p = 0.95
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
if dataset_name == "livecodebench/code_generation_lite":
    dataset = load_dataset(dataset_name, version_tag="release_v2", split="test", trust_remote_code=True)
else:
    dataset = load_dataset(dataset_name)
prompt_head = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\n\
You will be given a question (problem specification) and will generate a correct program that matches the specification and passes all tests. \
You will NOT return anything except for the program.\n\n\
Question: "
prompt_tail = "\n\n<|im_end|>\n<|im_start|>assistant\n"
def get_codeqwen_question_template_answer(question) -> str:
    # prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt = f"Question: {question["question_content"]}\n\n"
    if question["starter_code"]:
        prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{question["starter_code"]}\n```\n\n<|im_end|>\n"
    else:
        prompt += f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt

exec_batch = []
collect_res = []
if dataset_name == "livecodebench/code_generation_lite":
    for instance in tqdm(dataset):
        tail = get_codeqwen_question_template_answer(instance)
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\n\
You will be given a question (problem specification) and will generate a correct program that matches the specification and passes all tests. \
You will NOT return anything except for the program.\n\n" + tail
        exec_batch.append(prompt)
else:
    for instance in tqdm(dataset["train"]):
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

with open("collect_" + dataset_name.replace("/", "_") + ".json", "w") as f:
    json.dump(collect_res, f, indent=4)