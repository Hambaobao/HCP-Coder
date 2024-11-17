from typing import Dict


class DeepseekCoderTemplate:
    EOS_TOKEN: str = "<｜end▁of▁sentence｜>"
    SPLIT_TOKEN: str = ""
    STOP_TOKENS: list[str] = [
        "<｜fim▁begin｜>",
        "<｜fim▁hole｜>",
        "<｜fim▁end｜>",
        "<｜end▁of▁sentence｜>",
    ]

    @staticmethod
    def apply(item: Dict):
        if item["strategy"] in ['hcp', 'preliminary', 'random_all']:
            return DeepseekCoderTemplate.apply_hcp_template(item)
        elif item["strategy"] in ["rag"]:
            return DeepseekCoderTemplate.apply_rag_context_template(item)
        else:
            raise NotImplementedError

    @staticmethod
    def apply_hcp_template(item: Dict):
        context = item["context"]

        # build cross-file context for completion
        cross_file_prompt = f""
        cross_file_context = context["cross_file_context"]
        for file in cross_file_context:
            cross_file_prompt += f"# {file}\n{cross_file_context[file]}"

        # build in-file context for completion
        file = item["file"]
        infile_context = context["infile_context"]
        infile_prompt = f"# {file}\n<｜fim▁begin｜>{infile_context['prefix']}<｜fim▁hole｜>{infile_context['suffix']}<｜fim▁end｜>"

        return cross_file_prompt + infile_prompt

    @staticmethod
    def apply_rag_context_template(item: Dict):
        context = item["context"]
        prefix = item["prefix"]
        suffix = item["suffix"]

        prompt = f"{context}<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"

        return prompt


class Starcoder2Template:
    EOS_TOKEN: str = "<|endoftext|>"
    SPLIT_TOKEN: str = "<fim_middle>"
    STOP_TOKENS: list[str] = [
        "<repo_name>",
        "<file_sep>",
        "<fim_prefix>",
        "<fim_suffix>",
        "<fim_middle>",
        "<fim_pad>",
        "<|endoftext|>",
    ]

    @staticmethod
    def apply(item: Dict):
        if item["strategy"] in ['hcp', 'preliminary', 'random_all']:
            return Starcoder2Template.apply_hcp_template(item)
        elif item["strategy"] in ["rag"]:
            return Starcoder2Template.apply_rag_context_template(item)
        else:
            raise NotImplementedError

    @staticmethod
    def apply_hcp_template(item: Dict):
        context = item["context"]
        repo_name = context["repo_name"]

        # build cross-file context for completion
        cross_file_prompt = f"<repo_name>{repo_name}"
        cross_file_context = context["cross_file_context"]
        for file in cross_file_context:
            cross_file_prompt += f"<file_sep>{file}\n{cross_file_context[file]}"

        # build in-file context for completion
        file = item["file"]
        infile_context = context["infile_context"]
        infile_prompt = f"<file_sep>{file}<fim_prefix>{infile_context['prefix']}<fim_suffix>{infile_context['suffix']}<fim_middle>"

        return cross_file_prompt + infile_prompt

    @staticmethod
    def apply_rag_context_template(item: Dict):
        context = item["context"]
        prefix = item["prefix"]
        suffix = item["suffix"]

        prompt = f"{context}<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"

        return prompt


class CodeGemmaTemplate:
    EOS_TOKEN: str = "<eos>"
    SPLIT_TOKEN: str = ""
    STOP_TOKENS: list[str] = [
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|file_separator|>",
        "<eos>",
    ]

    @staticmethod
    def apply(item: Dict):
        if item["strategy"] in ['hcp', 'preliminary', 'random_all']:
            return CodeGemmaTemplate.apply_hcp_template(item)
        elif item["strategy"] in ["rag"]:
            return CodeGemmaTemplate.apply_rag_context_template(item)
        else:
            raise NotImplementedError

    @staticmethod
    def apply_hcp_template(item: Dict):
        context = item["context"]

        # build cross-file context for completion
        cross_file_prompt = f""
        cross_file_context = context["cross_file_context"]
        for file in cross_file_context:
            cross_file_prompt += f"<|file_separator|>{file}\n{cross_file_context[file]}"

        # build in-file context for completion
        file = item["file"]
        infile_context = context["infile_context"]
        infile_prompt = f"<|file_separator|>{file}<|fim_prefix|>{infile_context['prefix']}<|fim_suffix|>{infile_context['suffix']}<|fim_middle|>"

        return cross_file_prompt + infile_prompt

    @staticmethod
    def apply_rag_context_template(item: Dict):
        context = item["context"]
        prefix = item["prefix"]
        suffix = item["suffix"]

        prompt = f"{context}<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

        return prompt


class Qwen25CoderTemplate:
    EOS_TOKEN: str = "<|endoftext|>"
    SPLIT_TOKEN: str = ""
    STOP_TOKENS: list[str] = [
        "<|repo_name|>",
        "<|file_sep|>",
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|endoftext|>",
    ]

    @staticmethod
    def apply(item: Dict):
        if item["strategy"] in ['hcp', 'preliminary', 'random_all']:
            return Qwen25CoderTemplate.apply_hcp_template(item)
        elif item["strategy"] in ["rag"]:
            return Qwen25CoderTemplate.apply_rag_context_template(item)
        else:
            raise NotImplementedError

    @staticmethod
    def apply_hcp_template(item: Dict):
        context = item["context"]
        repo_name = context["repo_name"]

        # build cross-file context for completion
        cross_file_prompt = f"<|repo_name|>{repo_name}\n"
        cross_file_context = context["cross_file_context"]
        for file in cross_file_context:
            cross_file_prompt += f"<|file_sep|>{file}\n{cross_file_context[file]}\n"

        # build in-file context for completion
        file = item["file"]
        infile_context = context["infile_context"]
        infile_prompt = f"<|file_sep|>{file}<|fim_prefix|>{infile_context['prefix']}<|fim_suffix|>{infile_context['suffix']}<|fim_middle|>"

        return cross_file_prompt + infile_prompt

    @staticmethod
    def apply_rag_context_template(item: Dict):
        context = item["context"]
        prefix = item["prefix"]
        suffix = item["suffix"]

        prompt = f"{context}<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

        return prompt


TemplatesMapping = {
    "deepseek-coder-1.3b-base": DeepseekCoderTemplate,
    "deepseek-coder-6.7b-base": DeepseekCoderTemplate,
    "starcoder2-3b": Starcoder2Template,
    "starcoder2-7b": Starcoder2Template,
    "codegemma-2b": CodeGemmaTemplate,
    "codegemma-7b": CodeGemmaTemplate,
    "DeepSeek-Coder-V2-Lite-Base": DeepseekCoderTemplate,
    "Qwen2.5-Coder-0.5B": Qwen25CoderTemplate,
    "Qwen2.5-Coder-1.5B": Qwen25CoderTemplate,
    "Qwen2.5-Coder-3B": Qwen25CoderTemplate,
    "Qwen2.5-Coder-7B": Qwen25CoderTemplate,
    "Qwen2.5-Coder-14B": Qwen25CoderTemplate,
}
