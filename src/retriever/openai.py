import os
import tiktoken
import numpy as np

from typing import List
from openai import OpenAI, RateLimitError
from openai.types import Embedding
from dotenv import load_dotenv
from time import sleep

from src.node import FunctionNode

load_dotenv(override=True)


class OpenAIRetriever():

    def __init__(self,):
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('BASE_URL'),
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def cosine_similarity(a: List, b: List) -> float:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def create_embedding(self, text: str) -> Embedding:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002",
        )
        return response.data[0].embedding

    def create_embeddings(self, texts: List[str]) -> List[Embedding]:
        while True:
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model="text-embedding-ada-002",
                )
                return [d.embedding for d in response.data]
            except Exception as e:
                if isinstance(e, RateLimitError) and '429' in str(e):
                    print('> Rate limit exceeded. Waiting for 30 seconds...')
                    sleep(30)
                elif isinstance(e, KeyboardInterrupt):
                    raise e
                else:
                    sleep(5)
                    print('> Error:', e)

    def retrieve(
        self,
        query: str,
        nodes: List[FunctionNode],
        top_k: List[int] = [5],
        top_p: List[float] = [0.1],
        max_n_nodes: int = 300,
    ) -> List[List[FunctionNode]]:

        texts = [node.content for node in nodes]
        texts = [self.tokenizer.decode(self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})[:8191]) for text in texts]
        # if the number of nodes is too large, we need to split the texts into multiple parts
        if len(texts) > max_n_nodes:
            n_splits = len(texts) // max_n_nodes + 1
            texts = np.array_split(texts, n_splits)
            embeddings = []
            for text in texts:
                embeddings += self.create_embeddings(text)
        else:
            embeddings = self.create_embeddings(texts)
        query_embedding = self.create_embedding(query)

        scores = []
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
            node.score = OpenAIRetriever.cosine_similarity(query_embedding, embedding)
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
