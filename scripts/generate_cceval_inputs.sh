# generate inputs of preliminary study on Baseline Evaluation: random_all
python create_completion_metadata.py \
    --strategy random_all \
    --output_file inputs/crosscodeeval/preliminary/random_all.jsonl

# generate inputs of preliminary study on Baseline Evaluation: infile_only
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 0 \
    --d_level 0 \
    --output_file inputs/crosscodeeval/preliminary/infile_only.jsonl

# generate inputs of preliminary study on Topological Dependency Analysis: d_level=1
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 0 \
    --d_level 1 \
    --output_file inputs/crosscodeeval/preliminary/p_level.0.d_level.1.jsonl

# generate inputs of preliminary study on Topological Dependency Analysis: d_level=2
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 0 \
    --d_level 2 \
    --output_file inputs/crosscodeeval/preliminary/p_level.0.d_level.2.jsonl

# generate inputs of preliminary study on Topological Dependency Analysis: d_level=3
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 0 \
    --d_level 3 \
    --output_file inputs/crosscodeeval/preliminary/p_level.0.d_level.3.jsonl

# generate inputs of preliminary study on Topological Dependency Analysis: d_level=4
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 0 \
    --d_level 4 \
    --output_file inputs/crosscodeeval/preliminary/p_level.0.d_level.4.jsonl

# generate inputs of preliminary study on Topological Dependency Analysis: d_level=inf
# and inputs of preliminary study on Cross-File Content Analysis: p_level=0
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 0 \
    --d_level -1 \
    --output_file inputs/crosscodeeval/preliminary/p_level.0.d_level.inf.jsonl

# generate inputs of preliminary study on Cross-File Content Analysis: p_level=1
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 1 \
    --d_level -1 \
    --output_file inputs/crosscodeeval/preliminary/p_level.1.d_level.inf.jsonl

# generate inputs of preliminary study on Cross-File Content Analysis: p_level=2
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 2 \
    --d_level -1 \
    --output_file inputs/crosscodeeval/preliminary/p_level.2.d_level.inf.jsonl

# generate inputs of preliminary study on Cross-File Content Analysis: p_level=2 + d_level=1
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 2 \
    --d_level 1 \
    --output_file inputs/crosscodeeval/preliminary/p_level.2.d_level.1.jsonl

# generate inputs of preliminary study on Cross-File Content Analysis: p_level=2 + d_level=2
python create_completion_metadata.py \
    --strategy preliminary \
    --p_level 2 \
    --d_level 2 \
    --output_file inputs/crosscodeeval/preliminary/p_level.2.d_level.2.jsonl

# generate inputs of our HCP
python create_completion_metadata.py \
    --strategy hcp \
    --top_p 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --top_k 5 10 15 20 25 30 \
    --output_file inputs/crosscodeeval/hcp/hcp_all.jsonl

# split the HCP dataset by top_p and top_k
python utils/preprocess/split_hcp_all.py \
    --input_file inputs/crosscodeeval/hcp/hcp_all.jsonl \
    --output_dir inputs/crosscodeeval/hcp