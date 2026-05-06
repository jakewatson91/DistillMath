from datetime import datetime
import json
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.lm_styles import LanguageModel
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import build_prompt_benchmark,combine_results,sort_and_extract_save_results,get_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    # print("Cannot import vllm")
    pass
from lcb_runner.runner.base_runner import BaseRunner
class VLLMRunner(BaseRunner):
    def __init__(self, args, model_style, tokenizer_path, model_path):
        super().__init__(args, model_style)
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            enable_prefix_caching=args.enable_prefix_caching,
            trust_remote_code=args.trust_remote_code,
        )
        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=self.args.stop,
        )

    def _run_single(self, prompt: str) -> list[str]:
        pass

    def run_batch(self, prompts: list[str]) -> list[list[str]]:
        outputs = [None for _ in prompts]
        remaining_prompts = []
        remaining_indices = []
        for prompt_index, prompt in enumerate(prompts):
            if self.args.use_cache and prompt in self.cache:
                if len(self.cache[prompt]) == self.args.n:
                    outputs[prompt_index] = self.cache[prompt]
                    continue
            remaining_prompts.append(prompt)
            remaining_indices.append(prompt_index)
        if remaining_prompts:
            vllm_outputs = self.llm.generate(remaining_prompts, self.sampling_params)
            if self.args.use_cache:
                assert len(remaining_prompts) == len(vllm_outputs)
                for index, remaining_prompt, vllm_output in zip(
                    remaining_indices, remaining_prompts, vllm_outputs
                ):
                    self.cache[remaining_prompt] = [o.text for o in vllm_output.outputs]
                    outputs[index] = [o.text for o in vllm_output.outputs]
            else:
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [o.text for o in vllm_output.outputs]
        return outputs

def test_model(args, model_style: LanguageModel, tokenizer_path: str, model_path, eval_outpath: str):
    gen_file = eval_outpath.replace(".json", "_gen.json")
    eval_file = eval_outpath.replace(".json", "_eval.json")
    eval_all_file = eval_outpath.replace(".json", "_eval_all.json")
    # add timestamp to head
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_file = gen_file.replace(".json", f"_{timestamp}.json")
    eval_file = eval_file.replace(".json", f"_{timestamp}.json")
    eval_all_file = eval_all_file.replace(".json", f"_{timestamp}.json")

    benchmark, format_prompt = build_prompt_benchmark(args)
    if args.debug:
        print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:15]

    remaining_benchmark = benchmark

    runner = VLLMRunner(args, model_style, tokenizer_path, model_path)
    results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)

    combined_results = combine_results(args.scenario, results, model_style, args.cot_code_execution)

    save_results = [
            instance.insert_output(outputs_list, extracted_list)
            for instance, (outputs_list, extracted_list) in zip(
                remaining_benchmark, combined_results
            )
    ]
    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )
    with open(gen_file, "w") as f:
        json.dump(save_results, f, indent=4)

    metrics = get_metrics(args.scenario, args, benchmark, combined_results)
    graded = extract_instance_results(metrics[1])
    if metrics:
        metadatas = metrics[2]
    else:
        metadatas = [[] for _ in benchmark]
    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metadatas
        )
    ]
    with open(eval_file, "w") as f:
        json.dump(metrics, f, indent=4)
    with open(eval_all_file, "w") as f:
        json.dump(save_eval_results, f, indent=4)

if __name__ == "__main__":
    args = get_args()
    args.scenario = Scenario.codegeneration
    args.evaluation = True
    args.debug = False
    args.not_fast = False
    args.cot_code_execution = False
    args.tensor_parallel_size = 2
    args.start_date = "2024-07-01"
    args.end_date = "2024-11-30"
    model_style = LanguageModelStore["Qwen/Qwen2.5-Coder-32B-Instruct"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token  # causal LM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    # model.save_pretrained("/home/zhiyang/projects/distill/saved_model/model")
    # tokenizer.save_pretrained("/home/zhiyang/projects/distill/saved_model/tokenizer")
    # test_model(args, model_style, "/home/zhiyang/projects/distill/saved_model/tokenizer", "/home/zhiyang/projects/distill/saved_model/model", "/home/zhiyang/projects/distill/eval_report/eval.json")
    test_model(args, model_style, "Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct", "/home/zhiyang/projects/distill/eval_report/eval.json")