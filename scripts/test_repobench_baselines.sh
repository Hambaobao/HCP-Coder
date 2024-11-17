export VLLM_USE_MODELSCOPE=false

MODEL_NAME_OR_PATH=$1
TP=$2

MODEL_NAME=$(basename ${MODEL_NAME_OR_PATH})

run_completion_and_evaluation() {
    local INPUT_FILE=$1

    OUTPUT_DIR=outputs/repobench/baselines/${MODEL_NAME}/$(basename ${INPUT_FILE} .jsonl)

    python complete.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --tensor_parallel_size $TP \
        --input_file ${INPUT_FILE} \
        --output_file ${OUTPUT_DIR}/outputs.jsonl

    python evaluate.py \
        --input_file ${OUTPUT_DIR}/outputs.jsonl \
        --output_file ${OUTPUT_DIR}/results.json
}

# RAG ReAcc
run_completion_and_evaluation inputs/repobench/baselines/reacc_common.jsonl

# Infile Only
run_completion_and_evaluation inputs/repobench/baselines/infile_only_common.jsonl

# Random All
run_completion_and_evaluation inputs/repobench/baselines/random_all_common.jsonl

# Draco
case ${MODEL_NAME_OR_PATH} in
    *"codegemma"*)
        run_completion_and_evaluation inputs/repobench/baselines/draco_cg_common.jsonl
        ;;
    *"deepseek-coder"*)
        run_completion_and_evaluation inputs/repobench/baselines/draco_ds_common.jsonl
        ;;
    *"starcoder2"*)
        run_completion_and_evaluation inputs/repobench/baselines/draco_sc2_common.jsonl
        ;;
    *"DeepSeek-Coder-V2"*)
        run_completion_and_evaluation inputs/repobench/baselines/draco_dsv2_common.jsonl
        ;;
    *"Qwen2.5-Coder"*)
        run_completion_and_evaluation inputs/repobench/baselines/draco_qc_common.jsonl
        ;;
    *)
        echo "Draco baselines are not available for this model."
        ;;
esac