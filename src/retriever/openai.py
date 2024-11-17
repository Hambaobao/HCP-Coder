import os
import tiktoken

from typing import List
from openai import OpenAI, RateLimitError
from openai.types import Embedding
from dotenv import load_dotenv
from time import sleep

load_dotenv(override=True)


class OpenAIEmbeddings():

    def __init__(self,):
        self.client = OpenAI(
            base_url=os.getenv('OPENAI_BASE_URL'),
            api_key=os.getenv('OPENAI_API_KEY'),
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def create_embeddings(self, texts: List[str]) -> List[Embedding]:
        truncated_texts = [self.tokenizer.decode(self.tokenizer.encode(
            text,
            allowed_special={'<|endoftext|>'},
        )[:8191]) for text in texts]

        while True:
            try:
                response = self.client.embeddings.create(
                    input=truncated_texts,
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
