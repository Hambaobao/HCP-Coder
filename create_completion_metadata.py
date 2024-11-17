import json
import argparse

from typing import List
from tqdm import tqdm
from pathlib import Path

from src.topo import RepoTopo
from src.build_repo import build_repo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="crosscodeeval")
    parser.add_argument("--input_file", type=str, default="benchmarks/crosscodeeval/python_line_completion.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--d_level", type=int, default=0)
    parser.add_argument("--p_level", type=int, default=0)
    parser.add_argument("--strategy", type=str, default="hcp")
    parser.add_argument("--top_k", nargs="*", type=int, default=[5])
    parser.add_argument("--top_p", nargs="*", type=float, default=[0.1])

    return parser.parse_args()


def get_meta_info(
    benchmark: str,
    data: list,
    d_level: int,
    p_level: int,
    strategy: str,
    output_file: str,
    top_k: List[int] = [5],
    top_p: List[float] = [0.1],
):

    outputs = []
    for item in tqdm(data):
        if benchmark == "crosscodeeval":
            path_to_repo = build_repo(item)
        else:
            path_to_repo = item["path_to_real_repo"]
        try:
            repo_topo = RepoTopo(path_to_repo)
        except Exception as e:
            if e == KeyboardInterrupt:
                exit()
            else:
                print("Failed to build repo:", path_to_repo)
                continue

        try:
            context = repo_topo.get_completion_context(
                strategy=strategy,
                d_level=d_level,
                p_level=p_level,
                file_path=item["file"],
                row=item["hole_row_idx"],
                col=item["hole_col_idx"],
                top_k=top_k,
                top_p=top_p,
            )
        except Exception as e:
            print(e)
            continue

        outputs.append({
            "task_id": item["task_id"],
            "repository": item["repository"],
            "file": item["file"],
            "row": item["hole_row_idx"],
            "col": item["hole_col_idx"],
            "d_level": d_level,
            "strategy": strategy,
            "context": context,
            "groundtruth": item["groundtruth"],
        })

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
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
        args.benchmark,
        data,
        args.d_level,
        args.p_level,
        args.strategy,
        args.output_file,
        args.top_k,
        args.top_p,
    )

    print("> Total completed tasks:", len(meta_info) + len(history_data))
