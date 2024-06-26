python utils/auto/auto_generate.py \
    --data_file benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.1_topk_5.jsonl \
                benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.2_topk_5.jsonl \
                benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.3_topk_5.jsonl \
                benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.4_topk_5.jsonl \
                benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.5_topk_5.jsonl \
                benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.5_topk_5.jsonl \
                benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.6_topk_5.jsonl \
                benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.7_topk_5.jsonl \
                benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.8_topk_5.jsonl \
                benchmarks/crosscodeeval/hierarchy_topp_topk/topp_0.9_topk_5.jsonl \
    --truncate \
    --model_name_or_path ~/zoo/deepseek-ai/deepseek-coder-1.3b-base \
                        ~/zoo/deepseek-ai/deepseek-coder-6.7b-base \
    --cuda_device 2