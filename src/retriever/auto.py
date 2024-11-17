import numpy as np

from typing import List
from dotenv import load_dotenv

from ..node import FunctionNode
from .openai import OpenAIEmbeddings
from .jina import JinaEmbeddings

load_dotenv(override=True)


def cosine_similarity(a: List, b: List) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class AutoRetriever():

    def __init__(self, engine: str = "openai"):
        if engine == "openai":
            self.engine = OpenAIEmbeddings()
        elif engine == "jina":
            self.engine = JinaEmbeddings()
        else:
            raise ValueError("Unsupported engine")

    def retrieve(
        self,
        query: str,
        nodes: List[FunctionNode],
        top_k: List[int] = [5],
        top_p: List[float] = [0.1],
        max_n_nodes: int = 300,
    ) -> List[List[FunctionNode]]:

        texts = [node.content for node in nodes]
        # if the number of nodes is too large, we need to split the texts into multiple parts
        if len(texts) > max_n_nodes:
            n_splits = len(texts) // max_n_nodes + 1
            # chunk the texts into n_splits
            chunked_texts = np.array_split(texts, n_splits)
            embeddings = []
            for chunk in chunked_texts:
                embeddings += self.engine.create_embeddings(chunk)
        else:
            embeddings = self.engine.create_embeddings(texts)
        query_embedding = self.engine.create_embeddings([query])[0]

        scores = []
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
            node.score = cosine_similarity(query_embedding, embedding)
            scores.append(node.score)

        scores = np.array(scores)
        top_nodes = []
        for p in top_p:
            for k in top_k:
                border = np.quantile(scores, 1 - p)
                topp_indices = np.where(scores >= border)[0]
                topk_indices = topp_indices[np.argsort(scores[topp_indices])[-k:]]
                topk_nodes = [nodes[i] for i in topk_indices]
                topp_nodes = [nodes[i] for i in topp_indices]

                top_nodes.append({
                    'topk': k,
                    'topp': p,
                    'topk_nodes': topk_nodes,
                    'topp_nodes': topp_nodes,
                })

        return top_nodes
