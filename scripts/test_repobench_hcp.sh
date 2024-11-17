export VLLM_USE_MODELSCOPE=false

MODEL_NAME_OR_PATH=$1
TP=$2

MODEL_NAME=$(basename ${MODEL_NAME_OR_PATH})

run_completion_and_evaluation() {
    local INPUT_FILE=$1

    OUTPUT_DIR=outputs/repobench/hcp/${MODEL_NAME}/$(basename ${INPUT_FILE} .jsonl)

    python complete.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --tensor_parallel_size $TP \
        --input_file ${INPUT_FILE} \
        --output_file ${OUTPUT_DIR}/outputs.jsonl

    python evaluate.py \
        --input_file ${OUTPUT_DIR}/outputs.jsonl \
        --output_file ${OUTPUT_DIR}/results.json
}


# top-p=0.1
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.1_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.1_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.1_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.1_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.1_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.1_topk_30.jsonl

# top-p=0.2
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.2_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.2_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.2_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.2_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.2_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.2_topk_30.jsonl

# top-p=0.3
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.3_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.3_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.3_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.3_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.3_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.3_topk_30.jsonl

# top-p=0.4
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.4_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.4_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.4_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.4_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.4_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.4_topk_30.jsonl

# top-p=0.5
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.5_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.5_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.5_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.5_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.5_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.5_topk_30.jsonl

# top-p=0.6
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.6_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.6_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.6_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.6_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.6_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.6_topk_30.jsonl

# top-p=0.7
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.7_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.7_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.7_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.7_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.7_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.7_topk_30.jsonl

# top-p=0.8
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.8_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.8_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.8_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.8_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.8_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.8_topk_30.jsonl

# top-p=0.9
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.9_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.9_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.9_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.9_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.9_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_0.9_topk_30.jsonl

# top-p=1.0
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_1.0_topk_5.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_1.0_topk_10.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_1.0_topk_15.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_1.0_topk_20.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_1.0_topk_25.jsonl
run_completion_and_evaluation inputs/repobench/hcp/hcp_topp_1.0_topk_30.jsonl