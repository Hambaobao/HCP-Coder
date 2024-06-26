import os
import json
import argparse

from fuzzywuzzy import fuzz
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
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
    oom_count, passed_count, edit_sim, non_oom_sidt_sim, total = 0, 0, 0.0, 0.0, len(data)
    for d in data:
        pred, ref = d['response'].strip(), d['groundtruth'].strip()

        passed = True if pred == ref else False

        es = fuzz.ratio(pred, ref)

        outputs.append({
            'task_id': d['task_id'],
            'file': d['file'],
            'passed': passed,
            'response': d['response'],
            'groundtruth': d['groundtruth'],
            'edit_similarity': es,
        })

        if pred == 'OOM':
            oom_count += 1

        if passed:
            passed_count += 1
        edit_sim += es

        if pred != 'OOM':
            non_oom_sidt_sim += es

    print(f'> EM Accuracy: {passed_count / total * 100:.2f}%, EM Accuracy (w/o OOM): {passed_count / (total-oom_count) * 100:.2f}%, Edit Similarity: {edit_sim / total:.2f}, Edit Similarity (w/o OOM): {non_oom_sidt_sim / total:.2f}')

    return outputs


if __name__ == '__main__':

    args = parse_args()

    with open(args.input_file) as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]

    input_file = Path(args.input_file)

    print('-' * 60)
    print(f'> Evaluating {input_file.name}...')
    outputs = evaluate(data)

    output_path = os.path.join("results", input_file.name)
    with open(output_path, 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')
