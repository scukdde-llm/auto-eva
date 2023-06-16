import time
import openai
from abc import ABC, abstractclassmethod
from typing import Sequence

__all__ = [
    "Predictor",
    "OpenAIPredictor"
]


class Predictor(ABC):
    @abstractclassmethod
    def predict(self, query: str, history: Sequence[str] = []) -> str:
        pass


class OpenAIPredictor(Predictor):
    model_name_: str = "gpt-3.5-turbo"
    temperature_: float = 1
    top_p_: float = 1.0
    top_n_: int = 1
    presence_penalty_: float = 0.1
    frequency_penalty_: float = 0.1

    def __init__(self, **args) -> None:
        self.model_name_ = args["model_name"]
        self.temperature_ = args["temperature"]
        self.top_p_ = args["top_p"]
        self.top_n_ = args["top_n"]
        self.presence_penalty_ = args["presence_penalty"]
        self.frequency_penalty_ = args["frequency_penalty"]

    def predict(self, query: str, history: Sequence[str] = []) -> str:
        retry_cnt = 0
        retry_limit = 3

        while retry_cnt < retry_limit:
            try:
                chat_completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}],
                    temperature=self.temperature_,
                    top_p=self.top_p_,
                    n=self.top_n_,
                    presence_penalty=self.presence_penalty_,
                    frequency_penalty=self.frequency_penalty_
                )
                return chat_completion.choices[0].message.content
            except Exception as er:
                print(f"error {er}")
                retry_cnt += 1
                if retry_cnt >= retry_limit:
                    time.sleep(60)
