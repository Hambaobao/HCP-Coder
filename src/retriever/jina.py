import os

from typing import List
from dotenv import load_dotenv
from transformers import AutoModel
from pathlib import Path

load_dotenv(override=True)


class JinaEmbeddings():
    """
    This is a experimental feature, which uses Jina to create embeddings for the text.
    The correctness of the implementation is not guaranteed.
    """

    def __init__(self,):
        hf_models_home = os.getenv("HF_MODELS_HOME")
        self.model = AutoModel.from_pretrained(
            Path(hf_models_home) / "jinaai/jina-embeddings-v3",
            trust_remote_code=True,
        ).cuda()

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:

        embeddings = self.model.encode(texts, max_length=8191)
        return embeddings
