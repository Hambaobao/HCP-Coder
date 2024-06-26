import json
import numpy as np

starcoder2_files = [
    'outputs/dense_cross_dense_infile_truncate_false/starcoder2-3b_dense_cross_dense_infile_dep_0.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/starcoder2-3b_dense_cross_dense_infile_dep_1.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/starcoder2-3b_dense_cross_dense_infile_dep_2.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/starcoder2-3b_dense_cross_dense_infile_dep_3.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/starcoder2-3b_dense_cross_dense_infile_dep_4.jsonl',
    # 'outputs/dense_cross_dense_infile_truncate_false/starcoder2-3b_dense_cross_dense_infile_dep_all.jsonl',
]

codegemma_files = [
    'outputs/dense_cross_dense_infile_truncate_false/codegemma-2b_dense_cross_dense_infile_dep_0.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/codegemma-2b_dense_cross_dense_infile_dep_1.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/codegemma-2b_dense_cross_dense_infile_dep_2.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/codegemma-2b_dense_cross_dense_infile_dep_3.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/codegemma-2b_dense_cross_dense_infile_dep_4.jsonl',
    # 'outputs/dense_cross_dense_infile_truncate_false/starcoder2-3b_dense_cross_dense_infile_dep_all.jsonl',
]

deepseeker_files = [
    'outputs/dense_cross_dense_infile_truncate_false/deepseek-coder-1.3b-base_dense_cross_dense_infile_dep_0.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/deepseek-coder-1.3b-base_dense_cross_dense_infile_dep_1.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/deepseek-coder-1.3b-base_dense_cross_dense_infile_dep_2.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/deepseek-coder-1.3b-base_dense_cross_dense_infile_dep_3.jsonl',
    'outputs/dense_cross_dense_infile_truncate_false/deepseek-coder-1.3b-base_dense_cross_dense_infile_dep_4.jsonl',
    # 'outputs/dense_cross_dense_infile_truncate_false/deepseek-coder-1.3b-base_dense_cross_dense_infile_dep_all.jsonl',
]

for scf, cgf, dsf in zip(starcoder2_files, codegemma_files, deepseeker_files):
    with open(scf, 'r') as f:
        sc_data = [json.loads(line) for line in f]

    with open(cgf, 'r') as f:
        cg_data = [json.loads(line) for line in f]

    with open(dsf, 'r') as f:
        ds_data = [json.loads(line) for line in f]

    outputs = []
    for sc, cg, ds in zip(sc_data, cg_data, ds_data):
        assert sc['task_id'] == cg['task_id'], f"Task ID mismatch: {sc['task_id']} != {cg['task_id']}"
        assert sc['task_id'] == ds['task_id'], f"Task ID mismatch: {sc['task_id']} != {ds_data[0]['task_id']}"

        outputs.append({
            'prompt_length': sc['prompt_length'],
            'starcoder2_tokenized_length': sc['tokenized_prompt_length'],
            'codegemma_tokenized_length': cg['tokenized_prompt_length'],
            'deepseeker_tokenized_length': ds['tokenized_prompt_length'],
        })

    output_file = f"statistics/prompt_length/{scf.split('b_')[-1]}"
    with open(output_file, 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')

    starcoder2_lengths = []
    codegemma_lengths = []
    deepseek_lengths = []

    with open(output_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            starcoder2_lengths.append(data['starcoder2_tokenized_length'])
            codegemma_lengths.append(data['codegemma_tokenized_length'])
            deepseek_lengths.append(data['deepseeker_tokenized_length'])

    print('-' * 60)
    print(f"File: {output_file}")
    print(f"Starcoder2 median: {np.median(starcoder2_lengths)}, average: {int(np.mean(starcoder2_lengths))}")
    print(f"CodeGemma median: {np.median(codegemma_lengths)}, average: {int(np.mean(codegemma_lengths))}")
    print(f"DeepSeeker median: {np.median(deepseek_lengths)}, average: {int(np.mean(deepseek_lengths))}")
