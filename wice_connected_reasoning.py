from datasets import load_dataset
import json
from tqdm import tqdm
import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import random
from sklearn.metrics import balanced_accuracy_score
from factcg import GroundingScore
import os
import argparse
random.seed(2024)

def get_scores(scorer, method, premise_list, hypothesis_list):
    if method == "factcg":
        if "t5" in model_name_hf:
            raw_prob, _ = scorer.inference_example_real_batch(premise=premise_list, hypo=hypothesis_list)
        elif "llama" in model_name_hf:
            raw_prob, _ = scorer.inference_example_real_batch(premise=premise_list, hypo=hypothesis_list)
        else:
            raw_prob = scorer.score(contexts=premise_list, claims=hypothesis_list)
    elif method == "minicheck":
        pred_label, raw_prob, _, _ = scorer.score(docs=premise_list, claims=hypothesis_list)
    elif method == "alignscore":
        raw_prob = scorer.score(contexts=premise_list, claims=hypothesis_list)
    return raw_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run WiCE Connected Reasoning evaluation.')
    parser.add_argument('--method', type=str, choices=["factcg", "minicheck", "alignscore"], default="factcg", help='fact-checker to use')
    args = parser.parse_args()

    method = args.method
    assert method in ["factcg", "minicheck", "alignscore"]

    # load data
    with open("data/Wice_Connected_Reasoning/WiCE_CoRe.json", "r") as f:
        data = json.load(f)
        source_list = data["source"]
        corrupted_source_list = data["corrupted_source"]
        hypothesis_list = data["hypothesis"]

    assert len(source_list) == len(corrupted_source_list) == len(hypothesis_list)
    print("Number of data for CoRe test: ",len(source_list))

    if method == "factcg":
        ckpt = "ckpt/factcg_dbt.ckpt"
        model_name_hf = "microsoft/deberta-v3-large"

        prediction_head = "bin_sp"
        scorer = GroundingScore(model_name=model_name_hf, batch_size=16, ckpt_path=ckpt)
    elif method == "minicheck":
        from minicheck.minicheck import MiniCheck
        # scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./minicheck_ckpts')
        scorer = MiniCheck(model_name='deberta-v3-large', cache_dir='./minicheck_ckpts')
    elif method == "alignscore":
        from alignscore import AlignScore
        scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0', ckpt_path='/home/derenlei/AlignScore-large.ckpt', evaluation_mode='nli_sp')

    df_dev = pd.DataFrame(load_dataset("lytang/LLM-AggreFact")['dev'])
    sub_df = df_dev[df_dev.dataset == "Wice"]

    preds = get_scores(scorer, method, sub_df.doc.values, sub_df.claim.values)

    best_threshold = 0
    best_bacc_dev = 0

    for threshold in range(1, 100):
        binary_preds = [1 if p > threshold/100 else 0 for p in preds]
        bacc = balanced_accuracy_score(sub_df.label, binary_preds) * 100
        
        if bacc > best_bacc_dev:
            best_bacc_dev = bacc
            best_threshold = threshold/100

    print(f"Best threshold: {best_threshold}")
    print(f"Best Dev BAcc: {best_bacc_dev}")

    core_correct = 0
    higher = 0
    correct = 0
    preds = get_scores(scorer, method, source_list+corrupted_source_list, hypothesis_list+hypothesis_list)
    preds_pos = preds[:len(source_list)]
    preds_neg = preds[len(source_list):]
    preds_pos_bin = [1 if p > best_threshold else 0 for p in preds_pos]
    preds_neg_bin = [1 if p > best_threshold else 0 for p in preds_neg]
    for i in range(len(source_list)):
        if preds_pos_bin[i] == 1 and preds_neg_bin[i] == 0:
            core_correct += 1
        if preds_pos[i] > preds_neg[i]:
            higher += 1
        if preds_pos_bin[i] == 1:
            correct += 1
        if preds_neg_bin[i] == 0:
            correct += 1

    print(f"Accuracy CoRe: {core_correct/len(source_list)} ({core_correct}/{len(source_list)})")
    print(f"Precision CoRe: {core_correct/sum(preds_pos_bin)} ({core_correct}/{sum(preds_pos_bin)})")