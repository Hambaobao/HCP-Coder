from typing import List, Dict
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

import shutil
import json

REPOBENCH_RAW_DATA_ROOT = "benchmarks/repobench/raw_data"
REAL_REPO_DATA_ROOT = "benchmarks/repobench/real_repos"

import os


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):  # check if it is a file
                total_size += os.path.getsize(file_path)  # get file size
    return total_size / (1024 * 1024)


def recover_repository(item: Dict):
    repo_name = item["repo_name"]
    local_repo_path = Path(REPOBENCH_RAW_DATA_ROOT) / repo_name
    for code_file in item["repo_code"]:
        file_path = local_repo_path / code_file["path"]
        if file_path.exists():
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(code_file["code"])


def find_snippet_location(code: str, snippet: str):
    start_index = code.find(snippet)
    if start_index == -1:
        return None

    line_number = code[:start_index].count('\n') + 1

    last_newline_index = code.rfind('\n', 0, start_index)
    if last_newline_index == -1:
        column_number = start_index + 1
    else:
        column_number = start_index - last_newline_index

    assert column_number == 1, f"Column number is {column_number} instead of 1"

    return {"line_number": line_number, "column_number": column_number}


def construct_real_world_repository(id: int, item: Dict):
    repo_name = item["repo_name"]
    local_repo_path = Path(REPOBENCH_RAW_DATA_ROOT) / repo_name
    real_repo_path = Path(REAL_REPO_DATA_ROOT) / f"repobench-{id}" / repo_name
    file_path = local_repo_path / item["file_path"]
    real_repo_file_path = real_repo_path / item["file_path"]
    if not file_path.exists():
        return None

    folder_size = get_folder_size(local_repo_path)

    # skip large repositories
    if folder_size > 2:
        return None

    file_content = file_path.read_text()
    next_line = item["next_line"]
    location = find_snippet_location(file_content, next_line)

    # copy files from local_repo_path to real_repo_path
    shutil.copytree(local_repo_path, real_repo_path, dirs_exist_ok=True)

    code_lines = file_content.split('\n')

    assert code_lines[location["line_number"] - 1] == next_line, f"Code line does not match the groundtruth"

    code_lines[location["line_number"] - 1] = ""

    with open(real_repo_file_path, "w") as f:
        f.write('\n'.join(code_lines))

    location["path_to_real_repo"] = real_repo_path

    return location


if __name__ == "__main__":

    raw_ds = load_dataset("tianyang/repobench_raw_v1.1", split="python")

    # Recover the repositories
    for item in tqdm(raw_ds):
        recover_repository(item)

    ds = load_dataset("tianyang/repobench_python_v1.1", split="in_file")

    # construct real-world repositories
    outputs = []
    for id, item in enumerate(tqdm(ds)):
        try:
            location = construct_real_world_repository(id, item)
            if location is None:
                continue

            outputs.append({
                "task_id": f"repobench_{id}",
                "repository": item["repo_name"],
                "file": item["file_path"],
                "hole_row_idx": location["line_number"],
                "hole_col_idx": location["column_number"],
                "groundtruth": item["next_line"],
                "path_to_real_repo": str(location["path_to_real_repo"])
            })
        except Exception as e:
            print(f"Error in processing {item['repo_name']}")
            print(e)

    print(f"Total {len(outputs)} real-world repositories are constructed")

    output_file = "benchmarks/repobench/python_line_completion.jsonl"
    with open(output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")
