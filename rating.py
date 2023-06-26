import os
import sys
import json
import time
import openai
import pandas as pd

from typing import List, Dict, Tuple

query_prompt_ = """
你是一名给阅读理解题目评分的语文老师。你将得到一个问题、学生答案和正确答案，请为学生答案进行评分。
评分结果为0到100的整数，100分代表结果完全正确，0分代表结果错误，并解释为什么学生答案获得此得分。
仅根据学生的事实准确性对学生的答案进行评分。忽略学生答案和正确答案之间的标点符号和措辞差异。
如果学生答案包含的信息比真实答案多，只要它不包含任何相互冲突的陈述，就算正确。
如果学生回答上下文中没有提供特定信息，则答案不正确。
-------------------------------------------------------------------
例子一：
{"question": "《告全体党员书》是由哪一个组织发布的?",
 "correct answer": "中国国民党中央执行委员会",
 "student answer": "无法回答该问题。",
 "score": 0, 
 "reason": "学生完全无法回答问题，故得0分。"}
例子二：
{"question": "高射炮39有什么？",
 "correct answer": "220伏24千瓦发电机",
 "student answer": "高射炮39没有被提及。",
 "score": 0,
 "reason": "学生完全无法回答问题，故得0分。"}
例子三：
{"question": "为什么毛泽东拒绝停战？",
 "correct answer": "毛泽东认为志愿军有能力将联合国军逐出朝鲜半岛",
 "student answer": "为了逼迫史达林提供核弹技术，毛泽东拒绝停战。但苏联始终坚拒提供技术，毛泽东这才同意结束韩战，于1953年7月27日达成最后停战协议。",
 "score": 0,
 "reason": "学生未回答到要点，故得0分"}
例子四：
{"question": "葡萄牙糕点的两个例子是什么？",
 "correct answer": "来自里斯本的Pasteéis de Belém（或pastéis de nata）和来自Aveiro的ovos moles",
 "student answer": "Pasteéis de Belém和ovos moles是葡萄牙糕点的两个例子。",
 "score": 100,
 "reason": "回答涵盖关键要点，故得100分。"}
-------------------------------------------------------------------
请评阅以下题目和学生答案，严格遵守json格式：
{"question": "{question}",
 "correct answer": "{ref}", 
 "student answer": "{cand}",
 "score": ?,
 "reason": "?"}
"""


def openai_predict(query: str) -> str:
    retry_cnt = 0
    retry_limit = 3

    while retry_cnt < retry_limit:
        try:
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}],
                temperature=0.7,
                top_p=0.0,
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            return chat_completion.choices[0].message.content
        except Exception as er:
            print(f"error {er}")
            retry_cnt += 1
            if retry_cnt >= retry_limit:
                time.sleep(60)


def get_file_list(dir: str, endwith: str) -> List[str]:
    file_list = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(endwith):
                file_list.append(os.path.join(root, file))
    return file_list


def scroe_qa_data_by_gpt_3_0(question: str, ref: str, cand: str) -> Tuple[int, str]:
    """
        Returns:
            score, reason
    """
    ref = ref.replace('\n', '').replace('\r', '')
    cand = cand.replace('\n', '').replace('\r', '')

    query = query_prompt_.replace("{question}", question).replace(
        "{ref}", ref).replace("{cand}", cand)
    rsp = openai_predict(query=query)
    try:
        rsp = json.loads(rsp)
    except:
        print(rsp)
    return rsp["score"], rsp["reason"]


def score_qa_data(name: str, cands: List[Dict], refs: List[Dict]) -> Tuple[int, List[int]]:
    """
        Returns:
            avg_score, list[score]
    """
    case_size = len(cands)
    total_score = 0

    lQuestion: List[str] = []
    lScore: List[int] = []
    lReason: List[str] = []
    lRef_answer: List[str] = []
    lCand_answer: List[str] = []

    dfResult: pd.DataFrame = pd.DataFrame()

    for idx, _ in enumerate(cands):
        question = refs[idx]["q"]
        cand = cands[idx]["a"]
        ref = refs[idx]["a"]
        score, reason = scroe_qa_data_by_gpt_3_0(question, ref, cand)
        print(
            f"Question: {question}\n Ref: {ref}\n Ans: {cand}\nScore: {score} reason: {reason}")
        lReason.append(reason)
        lQuestion.append(question)
        lScore.append(score)
        lRef_answer.append(ref)
        lCand_answer.append(cand)

    dfResult["question"] = lQuestion
    dfResult["ref"] = lRef_answer
    dfResult["cand"] = lCand_answer
    dfResult["score"] = lScore
    dfResult["reason"] = lReason

    dfResult.to_csv(f"{name}.csv", index=False, sep=',')

    return total_score / case_size, lScore


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} [ref file path] [cand file dir]")
        exit(-1)

    ref_path = sys.argv[1]
    cand_dir_path = sys.argv[2]

    with open(ref_path, "r") as file:
        ref_data = json.loads(file.read())

    lRatingFile = get_file_list(dir=cand_dir_path, endwith='.json')

    score_data = pd.DataFrame()
    diff_file_and_question_score_data = pd.DataFrame()

    lFile: List[str] = []
    lAvgScore: List[int] = []
    for file in lRatingFile:
        with open(file, "r") as f:
            cand_data = json.loads(f.read())

        file_base_name = os.path.basename(file)
        avg_score, lScore = score_qa_data(
            name=file_base_name, cands=cand_data, refs=ref_data)

        lFile.append(file_base_name)
        lAvgScore.append(avg_score)
        diff_file_and_question_score_data[file_base_name] = lScore

    score_data["file"] = lFile
    score_data["score"] = lAvgScore

    score_data.to_csv("score_result.csv", index=False, sep=',')
