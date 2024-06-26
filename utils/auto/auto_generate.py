import os
import argparse
import subprocess

from pathlib import Path
from tqdm import tqdm

env = os.environ.copy()
env['TOKENIZERS_PARALLELISM'] = "false"

path_to_python_script = 'complete.py'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device', type=int, default=0, help='cuda device')
    parser.add_argument('--model_name_or_path', type=str, default=None, nargs='+', help='model name or path')
    parser.add_argument('--data_file', type=str, default=None, nargs='+', help='data file')
    parser.add_argument('--truncate', action='store_true', help='truncate data file')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    for model_name_or_path in args.model_name_or_path:
        for data_file in args.data_file:
            assert Path(data_file).exists(), f"Data file {data_file} does not exist!"
            model_name = model_name_or_path.split('/')[-1]
            output_file = Path("outputs") / f"{model_name}_{Path(data_file).name}"

            process_args = [
                '--model_name_or_path',
                model_name_or_path,
                '--data_file',
                str(data_file),
                '--output_file',
                str(output_file),
            ]

            if args.truncate:
                process_args.extend(['--truncate', 'True'])
            else:
                process_args.extend(['--truncate', 'False'])

            command = ['python', path_to_python_script] + process_args
            env['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

            run_cmd = ' '.join(command)
            print(f"> Running {Path(data_file).name} with {model_name} model")

            process = subprocess.Popen(command, env=env)

            process.wait()
