import torch
import json
import time

from tqdm import tqdm
from typing import List, Dict
from dataclasses import dataclass, field
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer

from utils.templates import TemplatesMapping

MAX_NEW_TOKENS = 32

MODEL_MAX_LENGTH_MAPPING = {
    "deepseek-coder": 16 * 1024 - MAX_NEW_TOKENS,
    "starcoder2": 16 * 1024 - MAX_NEW_TOKENS,
    "codegemma": 8 * 1024 - MAX_NEW_TOKENS,
}


@dataclass
class CompletionArguments:
    model_name_or_path: str = field(default="bigcode/starcoder2", metadata={"help": "The model checkpoint for weights initialization."})
    data_file: str = field(default="data.jsonl", metadata={"help": "The data file to use for training."})
    output_file: str = field(default="output.jsonl", metadata={"help": "The output file to save the results."})
    truncate: bool = field(default=True, metadata={"help": "Truncate the input to the model max length."})


def postprocess(template, raw_response: str) -> str:
    _raw_response = raw_response
    # this is solution to remove stop tokens for CodeGemma
    for stop_word in template.STOP_TOKENS:
        _raw_response = _raw_response.split(stop_word)[0]
    return _raw_response


def completion(
    data: List[Dict],
    model_name: str,
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    truncate: bool,
):

    template = TemplatesMapping[model_name]
    terminators = tokenizer.convert_tokens_to_ids(template.STOP_TOKENS)

    outputs = []
    for item in tqdm(data):
        prompt = template.apply(item)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=truncate,
        ).to(model.device)

        tokenized_prompt_length = len(inputs["input_ids"][0])
        num_new_tokens = 0

        start = time.time()
        try:
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,  # max number of tokens to generate
                do_sample=False,  # greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
            )
            num_new_tokens = len(output[0]) - tokenized_prompt_length
            raw_response = tokenizer.decode(output[0][tokenized_prompt_length:])
        except Exception as e:
            print(f"Error: {e}")
            raw_response = "OOM"
        end = time.time()

        # calculate the speed of the model
        time_elapsed = end - start
        token_per_second = num_new_tokens / time_elapsed

        # postprocess the response if nessasary
        response = postprocess(template, raw_response)

        outputs.append({
            "task_id": item["task_id"],
            "response": response,
            "raw_response": raw_response,
            "groundtruth": item["groundtruth"],
            "prompt_length": len(prompt),
            "tokenized_prompt_length": tokenized_prompt_length,
            "num_new_tokens": num_new_tokens,
            "time_elapsed": time_elapsed,
            "token_per_second": token_per_second,
            "prompt": prompt,
            "file": item["file"],
        })

    return outputs


if __name__ == "__main__":

    parser = HfArgumentParser((CompletionArguments))
    args = parser.parse_args_into_dataclasses()[0]

    model_names = ["starcoder2", "deepseek-coder", "codegemma"]
    model_name = next((name for name in model_names if name in args.model_name_or_path), None)
    assert model_name is not None, "Model name must be one of starcoder2, codegemma, or deepseek-coder"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        model_max_length=MODEL_MAX_LENGTH_MAPPING[model_name],
        truncation_side="left",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    model = torch.compile(model, mode='default')

    with open(args.data_file, "r") as f:
        data = [json.loads(line) for line in f]

    print("> Truncate:", args.truncate)
    outputs = completion(data, model_name, model, tokenizer, args.truncate)

    with open(args.output_file, "w") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")
