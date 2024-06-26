import json
import random
import argparse

random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="results/dense_cross_dense_infile_truncate_left/codegemma-7b_dense_cross_dense_infile_dep_all.jsonl")
    parser.add_argument('--output_file', type=str, default="statistics/bad_case/codegemma-7b.json")
    parser.add_argument('--num_samples', type=int, default=400, help="Number of samples to choose")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    data = [d for d in data if not d["passed"]]

    random.shuffle(data)
    cases = data[:args.num_samples]

    # sort by task_id
    cases = sorted(cases, key=lambda x: int(x["task_id"].split("/")[-1]))

    outputs = []
    for id, item in enumerate(cases):
        outputs.append({
            'id': id,
            **item,
        })

    # write to json file
    with open(args.output_file, "w") as f:
        json.dump(outputs, f, indent=4)

    print(f"Write to {args.output_file}")
