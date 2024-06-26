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
    --model_name_or_path ~/zoo/bigcode/starcoder2-3b \
                        ~/zoo/bigcode/starcoder2-7b \
    --cuda_device 1