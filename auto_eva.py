#!/bin/python3
from predictor import Predictor, OpenAIPredictor
from splitter import Splitter, CharacterTextSplitter
from irsystem import IRSystem, OpenAIEmbeddingIRSystem

import json
import toml

query_prompt = """
               使用中文从以下给出的文本（不使用任何超出文本以外的知识或者内容），回答问题：{question}。
               如果无法根据所给内容回答问题，请回复“无法回答该问题。”。
               -------------------------------------------------------------------
               {content}
               """


def make_predictor(**args) -> Predictor:
    llm_name = args["use"]
    if llm_name == "openai":
        return OpenAIPredictor(**args[llm_name])


def make_splitter(**args) -> Splitter:
    splitter_name = args["use"]
    if splitter_name == "text_splitter":
        return CharacterTextSplitter(**args[splitter_name])


def make_irsystem(splitter: Splitter, **args) -> IRSystem:
    ir_name = args["use"]
    if ir_name == "openai_embedding":
        return OpenAIEmbeddingIRSystem(splitter=splitter, **args[ir_name])


if __name__ == "__main__":
    data = None
    with open('auto_eva.toml', 'r') as file:
        data = toml.load(file)

    with open('./data/qa.json', 'r') as file:
        qa_data = json.loads(file.read())

    predictor = make_predictor(**data["llm"])
    splitter = make_splitter(**data["splitter"])
    irsystem = make_irsystem(splitter=splitter, **data["ir"])

    irsystem.build(["./data/docs"])

    eva_result = []
    for idx, qa in enumerate(qa_data):
        print(f"Auto eva: process question {idx+1}/{len(qa_data)}")
        question = qa["q"]
        content = irsystem.retrieval(question)
        query = query_prompt.replace(
            "{content}", content).replace("{question}", question)
        ans = predictor.predict(query=query)
        eva_result.append({"q": question, "a": ans})

    with open(f'{data["config"]["name"]}.json', 'w') as file:
        json.dump(eva_result, file)
