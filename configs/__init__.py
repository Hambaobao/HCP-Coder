from .templates import TemplatesMapping

MAX_NEW_TOKENS = 32

MODEL_MAX_LENGTH_MAPPING = {
    "deepseek-coder-1.3b-base": 16 * 1024,
    "deepseek-coder-6.7b-base": 16 * 1024,
    "starcoder2-3b": 16 * 1024,
    "starcoder2-7b": 16 * 1024,
    "codegemma-2b": 8 * 1024,
    "codegemma-7b": 8 * 1024,
    "DeepSeek-Coder-V2-Lite-Base": 32 * 1024,
    "Qwen2.5-Coder-0.5B": 32 * 1024,
    "Qwen2.5-Coder-1.5B": 32 * 1024,
    "Qwen2.5-Coder-3B": 32 * 1024,
    "Qwen2.5-Coder-7B": 32 * 1024,
    "Qwen2.5-Coder-14B": 32 * 1024,
}

SUPPORTED_MODELS = [
    "deepseek-coder-1.3b-base",
    "deepseek-coder-6.7b-base",
    "starcoder2-3b",
    "starcoder2-7b",
    "codegemma-2b",
    "codegemma-7b",
    "DeepSeek-Coder-V2-Lite-Base",
    "Qwen2.5-Coder-0.5B",
    "Qwen2.5-Coder-1.5B",
    "Qwen2.5-Coder-3B",
    "Qwen2.5-Coder-7B",
    "Qwen2.5-Coder-14B",
]
