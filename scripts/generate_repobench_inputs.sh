# preprocess the repobench dataset
python utils/preprocess/repobench.py

# generate inputs of Random-All
python create_completion_metadata.py \
    --benchmark repobench \
    --input_file benchmarks/repobench/python_line_completion.jsonl \
    --strategy random_all \
    --output_file inputs/repobench/baselines/random_all.jsonl

# generate inputs of our HCP
python create_completion_metadata.py \
    --benchmark repobench \
    --input_file benchmarks/repobench/python_line_completion.jsonl \
    --strategy hcp \
    --top_p 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --top_k 5 10 15 20 25 30 \
    --output_file inputs/repobench/hcp/hcp.jsonl


# split the HCP dataset by top_p and top_k
python utils/preprocess/split_hcp_all.py \
    --input_file inputs/repobench/hcp/hcp.jsonl \
    --output_dir inputs/repobench/hcp