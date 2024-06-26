import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_results_file', type=str, required=True)
    parser.add_argument('--new_results_file', type=str, required=True)
    return parser.parse_args()


def get_hit_changes(old_results, new_results):
    old_right = sum([1 for result in old_results if result['passed']])
    new_right = sum([1 for result in new_results if result['passed']])

    right_to_wrong, wrong_to_right = 0, 0
    for old, new in zip(old_results, new_results):
        if old['passed'] and not new['passed']:
            right_to_wrong += 1
        elif not old['passed'] and new['passed']:
            wrong_to_right += 1

    return old_right, new_right, right_to_wrong, wrong_to_right


if __name__ == "__main__":

    args = parse_args()

    with open(args.old_results_file, 'r') as f:
        old_results = [json.loads(line) for line in f]
    with open(args.new_results_file, 'r') as f:
        new_results = [json.loads(line) for line in f]

    old_right, new_right, right_to_wrong, wrong_to_right = get_hit_changes(old_results, new_results)

    print('-' * 60)
    print(f'> Old right: {old_right}, Right to wrong: {right_to_wrong}, Wrong to right: {wrong_to_right}, New right: {new_right}')
