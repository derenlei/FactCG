import argparse
from factcg import FactCGScore
from sklearn.metrics import balanced_accuracy_score
from datasets import load_dataset
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_scores(args, scorer, premise_list, hypothesis_list):
    if args.factcg:
        raw_prob = scorer.score(contexts=premise_list, claims=hypothesis_list)
    elif args.minicheck:
        pred_label, raw_prob, _, _ = scorer.score(
            docs=premise_list, claims=hypothesis_list)
    elif args.alignscore:
        raw_prob = scorer.score(contexts=premise_list, claims=hypothesis_list)
    elif args.summac_zs:
        raw_prob = scorer.score(premise_list, hypothesis_list)["scores"]
    elif args.summac_cv:
        raw_prob = scorer.score(premise_list, hypothesis_list)["scores"]
        assert len(raw_prob) == len(premise_list)
    return raw_prob


def get_threshold(args, scorer, devset_df):
    if args.threshold_setting == "fixed":
        print(f"Use fixed threshold 0.5")
        return 0.5
    elif args.threshold_setting == "tune":
        preds = get_scores(args, scorer, devset_df.doc.values,
                           devset_df.claim.values)

        best_threshold = 0
        best_bacc_dev = 0

        for threshold in range(1, 100):
            binary_preds = [1 if p > threshold/100 else 0 for p in preds]
            bacc = balanced_accuracy_score(devset_df.label, binary_preds) * 100

            if bacc > best_bacc_dev:
                best_bacc_dev = bacc
                best_threshold = threshold/100

        print(f"Best threshold for \
              {devset_df.dataset.unique().item()}: {best_threshold}")
        print(f"Best Dev BAcc for \
              {devset_df.dataset.unique().item()}: {best_bacc_dev}")
        return best_threshold


def run_testset(args, scorer, testset_df, threshold):
    preds = get_scores(
        args, scorer, testset_df.doc.values, testset_df.claim.values)
    binary_preds = [1 if p > threshold else 0 for p in preds]
    bacc_test = balanced_accuracy_score(
        testset_df.label, binary_preds) * 100
    return bacc_test


def run_benchmark(parser):
    args = parser.parse_args()

    if args.factcg:
        if not all((args.factcg_model_name, args.factcg_ckpt)):
            parser.error(
                '--factcg-model-name, --factcg-ckpt must be specified to run FactCG')
        scorer = FactCGScore(model_name=args.factcg_model_name,
                                batch_size=16, ckpt_path=args.factcg_ckpt)
    elif args.minicheck:
        if not all(args.minicheck_model_name):
            parser.error(
                '--minicheck-model-name must be specified to run MiniCheck')
        from minicheck.minicheck import MiniCheck
        scorer = MiniCheck(model_name=args.minicheck_model_name,
                           cache_dir='./minicheck_ckpts')
    elif args.alignscore:
        if not all((args.alignscore_ckpt)):
            parser.error(
                '--alignscore-ckpt must be specified to run AlignScore')
        from alignscore import AlignScore
        scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0',
                            ckpt_path=args.alignscore_ckpt, evaluation_mode='nli_sp')
    elif args.summac_zs:
        from summac.model_summac import SummaCZS
        scorer = SummaCZS(granularity="paragraph",
                          model_name="vitc", device="cuda")
    elif args.summac_cv:
        from summac.model_summac import SummaCConv
        scorer = SummaCConv(models=["vitc"], bins='percentile', granularity="paragraph",
                            nli_labels="e", device="cuda", start_file="default", agg="mean")

    else:
        parser.error('Method not supported.')

    devset_df = pd.DataFrame(load_dataset("lytang/LLM-AggreFact")['dev'])
    testset_df = pd.DataFrame(load_dataset("lytang/LLM-AggreFact")['test'])

    results = pd.DataFrame(columns=['Dataset', 'BAcc'])
    for dataset in devset_df.dataset.unique():
        # get the threshold
        sub_devset_df = devset_df[devset_df.dataset == dataset]
        threshold = get_threshold(args, scorer, sub_devset_df)

        # using the threshold to get the test results
        sub_testset_df = testset_df[testset_df.dataset == dataset]
        bacc_test = run_testset(args, scorer, sub_testset_df, threshold)
        results.loc[len(results)] = [dataset, bacc_test]
        print(f"Best Test BAcc for {dataset}: {bacc_test}")

    results.loc[len(results)] = ['Average', results.BAcc.mean()]
    results.round(1)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark against LLM-AggreFact")

    parser.add_argument('--threshold-setting', type=str, default="tune", choices=[
                        'fixed', 'tune'], help='Whether to use fixed threshold or dynamic')

    # parser group for FactCG
    factcg_parser = parser.add_argument_group('FactCG')
    factcg_parser.add_argument(
        '--factcg', action='store_true', help='Run FactCG against LLM-AggreFact')
    factcg_parser.add_argument('--factcg-model-name', type=str,
                               choices=['google/flan-t5-large', 'microsoft/deberta-v3-large'])
    factcg_parser.add_argument('--factcg-ckpt', type=str)

    # parser group for Minicheck
    minicheck_parser = parser.add_argument_group('Minicheck')
    minicheck_parser.add_argument(
        '--minicheck', action='store_true', help='Run Minicheck against LLM-AggreFact')
    minicheck_parser.add_argument('--minicheck-model-name', type=str,
                                  choices=['roberta-large', 'deberta-v3-large', 'flan-t5-large'])
    # parser group for AlignScore
    alignscore_parser = parser.add_argument_group('AlignScore')
    alignscore_parser.add_argument(
        '--alignscore', action='store_true', help='Run AlignScore against LLM-AggreFact')
    alignscore_parser.add_argument('--alignscore-ckpt', type=str)

    # parser group for summac_zs
    summac_zs_parser = parser.add_argument_group('SummaC-ZS')
    summac_zs_parser.add_argument(
        '--summac-zs', action='store_true', help='Run SummaC-ZS against LLM-AggreFact')

    # parser group for summac_cv
    summac_cv_parser = parser.add_argument_group('SummaC-CV')
    summac_cv_parser.add_argument(
        '--summac-cv', action='store_true', help='Run SummaC-CV against LLM-AggreFact')

    run_benchmark(parser)
