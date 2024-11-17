import json
import argparse

from typing import List, Dict
from vllm import LLM, SamplingParams, RequestOutput
from pathlib import Path

from configs import (
    SUPPORTED_MODELS,
    MODEL_MAX_LENGTH_MAPPING,
    TemplatesMapping,
    MAX_NEW_TOKENS,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    return parser.parse_args()


def do_complete(
    model: LLM,
    data: List[Dict],
    template,
    max_prompt_tokens: int,
):

    tokenizer = model.get_tokenizer()
    assert tokenizer.truncation_side == "left", "Truncation side must be left."
    stop_token_ids = tokenizer.convert_tokens_to_ids(template.STOP_TOKENS)

    prompts = [template.apply(item) for item in data]
    prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
    truncated_prompt_token_ids = [tokenizer.encode(
        prompt,
        truncation=True,
        max_length=max_prompt_tokens,
    ) for prompt in prompts]

    vllm_outputs = model.generate(
        prompt_token_ids=truncated_prompt_token_ids,
        sampling_params=SamplingParams(
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            top_p=1.0,
            stop=template.STOP_TOKENS,
            stop_token_ids=stop_token_ids,
        ),
    )

    completions = [item.outputs[0].text for item in vllm_outputs]

    def get_throughput(item: RequestOutput) -> int:
        num_tokens = len(item.outputs[0].token_ids)
        metrics = item.metrics
        generation_time = metrics.finished_time - metrics.arrival_time - metrics.scheduler_time
        throughput = num_tokens / generation_time
        return throughput

    throughputs = [get_throughput(item) for item in vllm_outputs]

    outputs = [{
        "task_id": item["task_id"],
        "completion": completion,
        "groundtruth": item["groundtruth"],
        "prompt_length": len(prompt),
        "num_prompt_tokens": len(prompt_token_id),
        "num_truncated_prompt_tokens": len(truncated_prompt_token_id),
        "throughput": throughput,
        "prompt": prompt,
        "file": item["file"],
    } for (
        item,
        prompt,
        completion,
        prompt_token_id,
        truncated_prompt_token_id,
        throughput,
    ) in zip(
        data,
        prompts,
        completions,
        prompt_token_ids,
        truncated_prompt_token_ids,
        throughputs,
    )]

    return outputs


if __name__ == "__main__":

    args = parse_args()

    model_name = args.model_name_or_path.split("/")[-1]

    assert model_name in SUPPORTED_MODELS, f"Model {model_name} is not supported."

    if Path(args.output_file).exists():
        print(f"Output file {args.output_file} already exists.")
        exit(0)

    model = LLM(
        args.model_name_or_path,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.98,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=MODEL_MAX_LENGTH_MAPPING[model_name],
        distributed_executor_backend="ray",
    )

    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    template = TemplatesMapping[model_name]

    max_prompt_tokens = MODEL_MAX_LENGTH_MAPPING[model_name] - MAX_NEW_TOKENS

    outputs = do_complete(model, data, template, max_prompt_tokens)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")
