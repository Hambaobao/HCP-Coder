export VLLM_USE_MODELSCOPE=false

MODEL_NAME_OR_PATH=$1
TP=$2

MODEL_NAME=$(basename ${MODEL_NAME_OR_PATH})

run_completion_and_evaluation() {
    local INPUT_FILE=$1

    OUTPUT_DIR=outputs/crosscodeeval/baselines/${MODEL_NAME}/$(basename ${INPUT_FILE} .jsonl)

    python complete.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --tensor_parallel_size $TP \
        --input_file ${INPUT_FILE} \
        --output_file ${OUTPUT_DIR}/outputs.jsonl

    python evaluate.py \
        --input_file ${OUTPUT_DIR}/outputs.jsonl \
        --output_file ${OUTPUT_DIR}/results.json
}

# RAG BM25
run_completion_and_evaluation inputs/crosscodeeval/baselines/python_rag_bm25.jsonl

# RAG OpenAI Cosine Similarity
run_completion_and_evaluation inputs/crosscodeeval/baselines/python_rag_openai_cosine_sim.jsonl

# RAG ReAcc
run_completion_and_evaluation inputs/crosscodeeval/baselines/python_rag_reacc.jsonl

# Draco
case ${MODEL_NAME_OR_PATH} in
    *"codegemma"*)
        run_completion_and_evaluation inputs/crosscodeeval/baselines/draco_codegemma.jsonl
        ;;
    *"deepseek-coder"*)
        run_completion_and_evaluation inputs/crosscodeeval/baselines/draco_deepseekcoder.jsonl
        ;;
    *"starcoder2"*)
        run_completion_and_evaluation inputs/crosscodeeval/baselines/draco_starcoder2.jsonl
        ;;
    *)
        echo "Draco baselines are not available for this model."
        ;;
esac