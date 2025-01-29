import argparse
import os
from factcg import GroundingScore
from sklearn.metrics import balanced_accuracy_score
from datasets import load_dataset
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser(description="Model Training Script")
parser.add_argument('--ckpt', type=str, required=True,
                    help='Path to the model checkpoint')
args = parser.parse_args()

ckpt = args.ckpt


df_dev = pd.DataFrame(load_dataset("lytang/LLM-AggreFact")['dev'])
df_test = pd.DataFrame(load_dataset("lytang/LLM-AggreFact")['test'])
method = "ours"
assert method in ["ours", "minicheck"]
setting = "tune"
assert setting in ["fixed", "tune"]


def get_scores(scorer, method, premise_list, hypothesis_list):
    if method == "ours":
        raw_prob = scorer.score(contexts=premise_list, claims=hypothesis_list)
    elif method == "minicheck":
        pred_label, raw_prob, _, _ = scorer.score(
            docs=premise_list, claims=hypothesis_list)
    return raw_prob


if method == "ours":
    model_name_hf = "microsoft/deberta-v3-large"
    scorer = GroundingScore(model_name=model_name_hf,
                            batch_size=16, ckpt_path=ckpt)
elif method == "minicheck":
    from minicheck.minicheck import MiniCheck
    scorer = MiniCheck(model_name='deberta-v3-large',
                       cache_dir='./minicheck_ckpts')


if setting == "fixed":
    result_df = pd.DataFrame(columns=['Dataset', 'BAcc'])
    threshold = 0.5
    for dataset in df_dev.dataset.unique():
        sub_df_test = df_test[df_test.dataset == dataset]
        # preds = scorer.score(contexts=sub_df_test.doc.values, claims=sub_df_test.claim.values)
        preds = get_scores(
            scorer, method, sub_df_test.doc.values, sub_df_test.claim.values)
        binary_preds = [1 if p > threshold else 0 for p in preds]
        bacc_test = balanced_accuracy_score(
            sub_df_test.label, binary_preds) * 100
        result_df.loc[len(result_df)] = [dataset, bacc_test]
        print(f"Best Test BAcc for {dataset}: {bacc_test}")

    result_df.loc[len(result_df)] = ['Average', result_df.BAcc.mean()]
    result_df.round(1)
    print(result_df)
elif setting == "tune":
    result_df = pd.DataFrame(columns=['Dataset', 'BAcc'])
    for dataset in df_dev.dataset.unique():
        sub_df = df_dev[df_dev.dataset == dataset]
        preds = get_scores(scorer, method, sub_df.doc.values,
                           sub_df.claim.values)

        best_threshold = 0
        best_bacc_dev = 0

        for threshold in range(1, 100):
            binary_preds = [1 if p > threshold/100 else 0 for p in preds]
            bacc = balanced_accuracy_score(sub_df.label, binary_preds) * 100

            if bacc > best_bacc_dev:
                best_bacc_dev = bacc
                best_threshold = threshold/100

        print(f"Best threshold for {dataset}: {best_threshold}")
        print(f"Best Dev BAcc for {dataset}: {best_bacc_dev}")

        sub_df_test = df_test[df_test.dataset == dataset]
        preds = get_scores(
            scorer, method, sub_df_test.doc.values, sub_df_test.claim.values)
        binary_preds = [1 if p > best_threshold else 0 for p in preds]
        bacc_test = balanced_accuracy_score(
            sub_df_test.label, binary_preds) * 100
        result_df.loc[len(result_df)] = [dataset, bacc_test]
        print(f"Best Test BAcc for {dataset}: {bacc_test}")

    result_df.loc[len(result_df)] = ['Average', result_df.BAcc.mean()]
    result_df.round(1)
    print(result_df)
