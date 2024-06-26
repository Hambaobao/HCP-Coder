import json
import argparse

from typing import List
from tqdm import tqdm
from pathlib import Path

from src.topo import RepoTopo
from utils.stimulate.build_repo import build_repo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="benchmarks/crosscodeeval/python_line_completion.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--dependency_level", type=int, default=0)
    parser.add_argument("--info_level", type=str, default="dense_cross_dense_infile")
    parser.add_argument("--top_k", nargs="*", type=int, default=[5])
    parser.add_argument("--top_p", nargs="*", type=float, default=[0.1])

    return parser.parse_args()


def get_meta_info(
    data: list,
    dependency_level: int,
    info_level: str,
    output_file: str,
    top_k: List[int] = [5],
    top_p: List[float] = [0.1],
):

    outputs = []
    for item in tqdm(data):
        path_to_repo = build_repo(item)
        try:
            repo_topo = RepoTopo(path_to_repo)
        except Exception as e:
            if e == KeyboardInterrupt:
                exit()
            else:
                print("Failed to build repo:", path_to_repo)
                continue

        context = repo_topo.get_completion_context(
            file_path=item["file"],
            row=item["hole_row_idx"],
            col=item["hole_col_idx"],
            dependency_level=dependency_level,
            info_level=info_level,
            top_k=top_k,
            top_p=top_p,
        )

        outputs.append({
            "task_id": item["task_id"],
            "repository": item["repository"],
            "file": item["file"],
            "row": item["hole_row_idx"],
            "col": item["hole_col_idx"],
            "dependency_level": dependency_level,
            "info_level": info_level,
            "context": context,
            "groundtruth": item["groundtruth"],
        })

        with open(output_file, "a") as f:
            f.write(json.dumps(outputs[-1]) + "\n")

    print("> Finished processing all tasks.")
    return outputs


def resume_from_history(file_path: str):
    if not Path(file_path).exists():
        return []
    print("> Resuming from history data.")
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]
    print("> Resumed tasks:", len(data))
    return data


if __name__ == "__main__":

    args = parse_args()
    with open(args.input_file, "r") as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]

    history_data = resume_from_history(args.output_file)
    data = data[len(history_data):]

    meta_info = get_meta_info(
        data,
        args.dependency_level,
        args.info_level,
        args.output_file,
        args.top_k,
        args.top_p,
    )

    print("> Total completed tasks:", len(meta_info) + len(history_data))
