from splitter import Splitter

import os
import faiss
import openai
import threading
import numpy as np
from abc import ABC, abstractclassmethod
from typing import List, Dict


__all__ = [
    "IRSystem",
    "OpenAIEmbeddingIRSystem"
]


def get_file_list(dir: str) -> List[str]:
    file_list = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.txt'):
                file_list.append(os.path.join(root, file))
    return file_list


class IRSystem(ABC):
    splitter_: Splitter = None

    @abstractclassmethod
    def build(self, dir: List[str]):
        pass

    @abstractclassmethod
    def retrieval(self, query: str) -> str:
        pass


class OpenAIEmbeddingIRSystem(IRSystem):
    model_name_: str = "text-embedding-ada-002"
    top_k_: int = 1
    index_: faiss = None
    idx_: int = 0
    idx2chunk: Dict[int, str] = {}
    lock_ = threading.Lock()

    def __init__(self, splitter: Splitter, **args) -> None:
        self.splitter_ = splitter
        self.top_k_ = args["top_k"]
        self.model_name_ = args["model_name"]
        self.index_ = faiss.IndexFlatL2(1536)

    def _get_embedding(self, text: str) -> List[float]:
        return openai.Embedding.create(input=[text], model=self.model_name_)['data'][0]['embedding']

    def _build_thread_task(self, rstart: int, rend: int, chunk_list: List[str]):
        for i in range(rstart, rend):
            chunk = chunk_list[i]
            embedding = self._get_embedding(chunk)
            self.lock_.acquire()
            print(f"IRSystem: process {self.idx_ + 1}/{len(chunk_list)}")
            self.index_.add(np.array([embedding], dtype=np.float32))
            self.idx2chunk[self.idx_] = chunk
            self.idx_ = self.idx_ + 1
            self.lock_.release()

    def retrieval(self, query: str) -> str:
        embedding = self._get_embedding(query)
        _, idxs = self.index_.search(np.array([embedding]), self.top_k_)
        content = ""
        for i in range(0, self.top_k_):
            content += (self.idx2chunk[idxs[0][i]] + "\n")
        return content

    def build(self, dir: List[str]):
        file_list = []
        chunk_list = []

        for d in dir:
            file_list += get_file_list(d)

        for f in file_list:
            with open(f, "r") as file:
                file_content = file.read()
            tmp_chunk_list = self.splitter_.split_text(file_content)
            chunk_list += tmp_chunk_list

        threads: List[threading.Thread] = []
        chunk_len = len(chunk_list)
        i = 0
        while (i < chunk_len):
            rend = i + 200
            if rend > chunk_len:
                rend = chunk_len
            task = threading.Thread(
                target=self._build_thread_task, args=(i, rend, chunk_list,))
            threads.append(task)
            task.start()
            i = i + 200

        for t in threads:
            t.join()
