import json
import argparse

from fuzzywuzzy import fuzz
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=False)
    return parser.parse_args()


def cal_edit_sim(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        edit_sim += fuzz.ratio(pred, gt)
    return edit_sim / total


def evaluate(data):
    outputs = []
    throughput = 0.0
    passed_count, edit_sim, total = 0, 0.0, len(data)
    for d in data:
        pred, ref = d['completion'].strip(), d['groundtruth'].strip()

        passed = True if pred == ref else False

        es = fuzz.ratio(pred, ref)

        outputs.append({
            'task_id': d['task_id'],
            'file': d['file'],
            'passed': passed,
            'completion': d['completion'],
            'groundtruth': d['groundtruth'],
            'edit_similarity': es,
            'throughput': d['throughput'],
        })

        throughput += d['throughput']

        if passed:
            passed_count += 1
        edit_sim += es

    results = {
        "EM": 100 * passed_count / total,
        "ES": edit_sim / total,
        "Throughput": throughput / total,
        "Details": outputs,
    }

    return results


if __name__ == '__main__':

    args = parse_args()

    with open(args.input_file) as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]

    input_file = Path(args.input_file)

    print('-' * 60)
    print(f'> Evaluating {input_file.name}...')
    results = evaluate(data)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
