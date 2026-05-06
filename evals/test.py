import os
from time import sleep
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # causal LM

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
    print(model)

    # --- Example prompts ---
    prompts = [
        "Write a Python function to compute factorial",
        "Explain pipeline parallelism in deep learning",
        "Summarize the book 'The Hobbit' in one paragraph"
    ]

    # --- Tokenize inputs ---
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(f'cuda:0')  # send to GPU 0 for the first stage

    # --- Generate text ---
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # number of new tokens to generate
            do_sample=True,      # enable randomness
            temperature=0.7,     # randomness control
            top_p=0.9            # nucleus sampling
        )

    # --- Decode outputs ---
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # --- Print results ---
    for prompt, text in zip(prompts, generated_texts):
        print(f"\nPrompt: {prompt}\nGenerated: {text}\n")
    
    sleep(10)  # keep the process alive to inspect GPU memory


if __name__ == "__main__":
    main()
