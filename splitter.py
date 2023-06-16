from typing import List
from abc import ABC, abstractclassmethod
from langchain.text_splitter import CharacterTextSplitter as LangChainTextSplitter

__all__ = [
    "Splitter",
    "CharacterTextSplitter"
]


class Splitter(ABC):
    @abstractclassmethod
    def split_text(self, text: str) -> List[str]:
        pass


class CharacterTextSplitter(Splitter):
    splitter_: LangChainTextSplitter = None

    def __init__(self, **args) -> None:
        self.splitter_ = LangChainTextSplitter(
            separator="\n",
            chunk_size=args["chunk_size"],
            chunk_overlap=args["chunk_overlap"]
        )

    def split_text(self, text: str) -> List[str]:
        return self.splitter_.split_text(text)
