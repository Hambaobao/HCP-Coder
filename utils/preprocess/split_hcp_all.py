import json
import copy
import argparse

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    with open(args.input_file, "r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]

    topps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    topks = [5, 10, 15, 20, 25, 30]

    for p in topps:
        for k in topks:
            print(f"> Processing topp: {p}, topk: {k}")
            output_file = f"{args.output_dir}/hcp_topp_{p}_topk_{k}.jsonl"
            outputs = []
            for d in tqdm(data):
                temp = copy.deepcopy(d)
                temp['context']['cross_file_context'] = d['context']['cross_file_context'][f"topp_{p}_topk_{k}"]
                outputs.append(temp)
            with open(output_file, "w") as f:
                for o in outputs:
                    f.write(json.dumps(o) + "\n")
