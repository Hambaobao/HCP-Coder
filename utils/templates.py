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
        if item["info_level"] in [
                "dense_cross_dense_infile",
                "concise_cross_dense_infile",
                "sparse_cross_dense_infile",
                "dense_random_cross_dense_infile",
                "hierarchical_cross_dense_infile",
        ]:
            return DeepseekCoderTemplate.apply_cross_file_completion_template(item)
        elif item["info_level"] in ["rag"]:
            return DeepseekCoderTemplate.apply_rag_context_template(item)
        else:
            raise NotImplementedError

    @staticmethod
    def apply_cross_file_completion_template(item: Dict):
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
        if item["info_level"] in [
                "dense_cross_dense_infile",
                "concise_cross_dense_infile",
                "sparse_cross_dense_infile",
                "dense_random_cross_dense_infile",
                "hierarchical_cross_dense_infile",
        ]:
            return Starcoder2Template.apply_cross_file_completion_template(item)
        elif item["info_level"] in ["rag"]:
            return Starcoder2Template.apply_rag_context_template(item)
        else:
            raise NotImplementedError

    @staticmethod
    def apply_cross_file_completion_template(item: Dict):
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
        if item["info_level"] in [
                "dense_cross_dense_infile",
                "concise_cross_dense_infile",
                "sparse_cross_dense_infile",
                "dense_random_cross_dense_infile",
                "hierarchical_cross_dense_infile",
        ]:
            return CodeGemmaTemplate.apply_cross_file_completion_template(item)
        elif item["info_level"] in ["rag"]:
            return CodeGemmaTemplate.apply_rag_context_template(item)
        else:
            raise NotImplementedError

    @staticmethod
    def apply_cross_file_completion_template(item: Dict):
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


TemplatesMapping = {
    "deepseek-coder": DeepseekCoderTemplate,
    "starcoder2": Starcoder2Template,
    "codegemma": CodeGemmaTemplate,
}
