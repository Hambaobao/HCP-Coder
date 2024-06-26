import os
import argparse
import subprocess

from pathlib import Path
from tqdm import tqdm

env = os.environ.copy()

path_to_evaluate_script = 'evaluate.py'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, nargs='+', help='Input directories')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    jsonl_files = [file for input_dir in args.input_dir for file in Path(input_dir).glob("**/*.jsonl")]

    print(f"> Available model outputs:")
    for id, jsonl_file in enumerate(jsonl_files):
        print(f"> [{id}]: {jsonl_file}")

    for jsonl_file in tqdm(jsonl_files):

        # Evaluate
        command = [
            'python',
            path_to_evaluate_script,
            "--input_file",
            jsonl_file,
        ]

        print(f"> Evaluating {jsonl_file} ...")
        process = subprocess.Popen(command, env=env)

        process.wait()
