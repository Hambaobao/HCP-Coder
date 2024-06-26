import json

from tqdm import tqdm


def process_python_line_completion():

    def extract_info(item: dict):
        task_id = item["metadata"]["task_id"]
        repository = item["metadata"]["repository"]
        file = item["metadata"]["file"]

        prefix = item["prompt"]
        lines = prefix.split("\n")
        hole_line = lines[-1]
        hole_row_idx = len(lines)
        hole_col_idx = len(hole_line)

        groundtruth = item["groundtruth"]

        return {
            "task_id": task_id,
            "repository": repository,
            "file": file,
            "hole_row_idx": hole_row_idx,
            "hole_col_idx": hole_col_idx,
            "groundtruth": groundtruth,
        }

    input_file = "benchmarks/crosscodeeval/data/python/line_completion.jsonl"
    with open(input_file, "r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]

    outputs = []
    for item in tqdm(data):
        outputs.append(extract_info(item))

    output_file = "benchmarks/crosscodeeval/python_line_completion.jsonl"
    with open(output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")


def process_python_rag_completion(
    input_file="benchmarks/crosscodeeval/data/python/line_completion_rg1_bm25.jsonl",
    output_file="benchmarks/crosscodeeval/python_rag_bm25.jsonl",
):

    def extract_info(item: dict):
        task_id = item["metadata"]["task_id"]
        repository = item["metadata"]["repository"]
        file = item["metadata"]["file"]

        prefix = item["prompt"]
        suffix = item["right_context"]
        context = item["crossfile_context"]["text"]
        groundtruth = item["groundtruth"]

        return {
            "task_id": task_id,
            "info_level": "rag",
            "repository": repository,
            "file": file,
            "groundtruth": groundtruth,
            "prefix": prefix,
            "suffix": suffix,
            "context": context,
        }

    with open(input_file, "r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]

    outputs = []
    for item in tqdm(data):
        outputs.append(extract_info(item))

    with open(output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":

    process_python_line_completion()

    process_python_rag_completion()
