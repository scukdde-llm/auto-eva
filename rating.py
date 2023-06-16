import json
import bert_score

if __name__ == "__main__":
    scorer = bert_score.BERTScorer(lang="zh", batch_size=3)

    with open("./config/auto_eva_t_2_0.json", "r") as file:
        case0_data = json.loads(file.read())

    with open("./test_data/qa/qa.json", "r") as file:
        case1_data = json.loads(file.read())

    sum_p = 0
    sum_r = 0
    sum_f1 = 0
    sum_e = 0
    total_size = len(case0_data)
    for idx, _ in enumerate(case0_data):
        case0 = case0_data[idx]["a"]
        case1 = case1_data[idx]["a"]
        if case0 == None or len(case0) <= 1 or "无法回答" in case0:
            sum_e += 1
        if case0 == None:
            case0 = ""
        print(f"Ans: {case0} / {case1}")
        (P, R, F1) = scorer.score([case0], [case1], return_hash=False)
        sum_p += P.mean().item()
        sum_r += R.mean().item()
        sum_f1 += F1.mean().item()
        print(f"Precision = {P.mean()}, Recall = {R.mean()}, F1 = {F1.mean()}")

    avg_p = sum_p / total_size
    avg_r = sum_r / total_size
    avg_f1 = sum_f1 / total_size
    avg_c = (total_size - sum_e) / total_size

    print(
        f"Precision = {avg_p}, Recall = {avg_r}, F1 = {avg_f1}, Correct = {avg_c}")
