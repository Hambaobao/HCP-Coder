export VLLM_USE_MODELSCOPE=false

MODEL_NAME_OR_PATH=$1
TP=$2

MODEL_NAME=$(basename ${MODEL_NAME_OR_PATH})

run_completion_and_evaluation() {
    local INPUT_FILE=$1

    OUTPUT_DIR=outputs/crosscodeeval/preliminary/${MODEL_NAME}/$(basename ${INPUT_FILE} .jsonl)

    python complete.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --tensor_parallel_size $TP \
        --input_file ${INPUT_FILE} \
        --output_file ${OUTPUT_DIR}/outputs.jsonl

    python evaluate.py \
        --input_file ${OUTPUT_DIR}/outputs.jsonl \
        --output_file ${OUTPUT_DIR}/results.json
}

# Infile only
run_completion_and_evaluation inputs/crosscodeeval/preliminary/infile_only.jsonl

# D-Level: 1
run_completion_and_evaluation inputs/crosscodeeval/preliminary/p_level.0.d_level.1.jsonl

# D-Level: 2
run_completion_and_evaluation inputs/crosscodeeval/preliminary/p_level.0.d_level.2.jsonl

# D-Level: 3
run_completion_and_evaluation inputs/crosscodeeval/preliminary/p_level.0.d_level.3.jsonl

# D-Level: 4
run_completion_and_evaluation inputs/crosscodeeval/preliminary/p_level.0.d_level.4.jsonl

# D-Level: inf
run_completion_and_evaluation inputs/crosscodeeval/preliminary/p_level.0.d_level.inf.jsonl

# P-Level: 1
run_completion_and_evaluation inputs/crosscodeeval/preliminary/p_level.1.d_level.inf.jsonl

# P-Level: 2
run_completion_and_evaluation inputs/crosscodeeval/preliminary/p_level.2.d_level.inf.jsonl

# P-Level: 2 + D-Level: 1
run_completion_and_evaluation inputs/crosscodeeval/preliminary/p_level.2.d_level.1.jsonl

# P-Level: 2 + D-Level: 2
run_completion_and_evaluation inputs/crosscodeeval/preliminary/p_level.2.d_level.2.jsonl