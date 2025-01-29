from pytorch_lightning import Trainer, seed_everything
from factcg.dataloader import AlignmentDataLoader
from factcg.grounding_model import GroundingModelForMultitaskLearning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
import os
import torch
#remove user warning
import warnings
warnings.simplefilter(action='ignore')

def train(datasets, args):
    dm = AlignmentDataLoader(
        dataset_config=datasets, 
        model_name=args.model_name, 
        sample_mode='seq',
        train_batch_size=args.batch_size,
        eval_batch_size=16,
        num_workers=args.num_workers, 
        train_eval_split=0.95,
        need_mlm=args.do_mlm
    )
    dm.setup()

    if args.ckpt_path != "":
        model = GroundingModelForMultitaskLearning.load_from_checkpoint(
            args.ckpt_path,
            model_name=args.model_name,
            adam_epsilon=args.adam_epsilon,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps_portion=args.warm_up_proportion
        )
    else:
        model = GroundingModelForMultitaskLearning(
            model_name=args.model_name,
            adam_epsilon=args.adam_epsilon,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps_portion=args.warm_up_proportion
        )

    model.need_mlm = args.do_mlm

    checkpoint_name = f"{args.ckpt_comment}{args.model_name.replace('/', '-')}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_save_path,
        filename=checkpoint_name + "_{epoch:02d}_{step}",
        every_n_train_steps=5000,
        monitor="train_loss",
        save_top_k=5
    )

    if "t5" in args.model_name.lower():
        precision = 32
    else:
        precision = 16

    logger = TensorBoardLogger("logs", name=args.ckpt_save_path.replace("/", "").replace(".", ""))
    trainer = Trainer(
        accelerator='gpu', 
        max_epochs=args.num_epoch, 
        devices=args.devices, 
        strategy="ddp_find_unused_parameters_true", 
        precision=precision,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batch,
        logger = logger
    )
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(os.path.join(args.ckpt_save_path, f"{checkpoint_name}_final.ckpt"))
    print("Training is finished.")

if __name__ == "__main__":
    # Data at BizQA08, BizQA10 /home/derenlei/GroundingModel/data/training/
    ALL_TRAINING_DATASETS = {
        ### NLI
        'mnli': {'task_type': 'nli', 'data_path': 'mnli.json'},     
        'doc_nli': {'task_type': 'bin_nli', 'data_path': 'doc_nli.json'},
        'snli': {'task_type': 'nli', 'data_path': 'snli.json'},
        'anli_r1': {'task_type': 'nli', 'data_path': 'anli_r1.json'},
        'anli_r2': {'task_type': 'nli', 'data_path': 'anli_r2.json'},
        'anli_r3': {'task_type': 'nli', 'data_path': 'anli_r3.json'},
        # 'mnli': {'task_type': 'nli_to_bin', 'data_path': 'mnli.json'},     
        # 'snli': {'task_type': 'nli_to_bin', 'data_path': 'snli.json'},
        # 'anli_r1': {'task_type': 'nli_to_bin', 'data_path': 'anli_r1.json'},
        # 'anli_r2': {'task_type': 'nli_to_bin', 'data_path': 'anli_r2.json'},
        # 'anli_r3': {'task_type': 'nli_to_bin', 'data_path': 'anli_r3.json'},

        ### fact checking
        'nli_fever': {'task_type': 'fact_checking', 'data_path': 'nli_fever.json'},
        'vitaminc': {'task_type': 'fact_checking', 'data_path': 'vitaminc.json'},
        # 'nli_fever': {'task_type': 'fact_checking_to_bin', 'data_path': 'nli_fever.json'},
        # 'vitaminc': {'task_type': 'fact_checking_to_bin', 'data_path': 'vitaminc.json'},

        ### paraphrase
        'paws': {'task_type': 'paraphrase', 'data_path': 'paws.json'},
        # 'paws_qqp': {'task_type': 'paraphrase', 'data_path': 'paws_qqp.json'},
        'paws_unlabeled': {'task_type': 'paraphrase', 'data_path': 'paws_unlabeled.json'},
        'qqp': {'task_type': 'paraphrase', 'data_path': 'qqp.json'},
        # 'wiki103': {'task_type': 'paraphrase', 'data_path': 'wiki103.json'},

        ### QA
        'squad_v2': {'task_type': 'qa', 'data_path': 'squad_v2_new.json'},
        'race': {'task_type': 'qa', 'data_path': 'race.json'},
        'adversarial_qa': {'task_type': 'qa', 'data_path': 'adversarial_qa.json'},
        'drop': {'task_type': 'qa', 'data_path': 'drop.json'},
        'hotpot_qa_distractor': {'task_type': 'qa', 'data_path': 'hotpot_qa_distractor.json'},
        'hotpot_qa_fullwiki': {'task_type': 'qa', 'data_path': 'hotpot_qa_fullwiki.json'},
        'newsqa': {'task_type': 'qa', 'data_path': 'newsqa.json'},
        'quoref': {'task_type': 'qa', 'data_path': 'quoref.json'},
        'ropes': {'task_type': 'qa', 'data_path': 'ropes.json'},
        'boolq': {'task_type': 'qa', 'data_path': 'boolq.json'},
        'eraser_multi_rc': {'task_type': 'qa', 'data_path': 'eraser_multi_rc.json'},
        'quail': {'task_type': 'qa', 'data_path': 'quail.json'},
        'sciq': {'task_type': 'qa', 'data_path': 'sciq.json'},
        # 'strategy_qa': {'task_type': 'qa', 'data_path': 'strategy_qa.json'},

        ### Coreference
        'gap': {'task_type': 'coreference', 'data_path': 'gap.json'}, # drop?

        ### Summarization
        # 'wikihow': {'task_type': 'summarization', 'data_path': 'wikihow.json'},
        # 'samsum': {'task_type': 'summarization', 'data_path': 'samsum.json'},

        ### Information Retrieval
        'msmarco': {'task_type': 'ir', 'data_path': 'msmarco.json'},

        ### STS
        # 'stsb': {'task_type': 'sts', 'data_path': 'stsb.json'},
        # 'sick': {'task_type': 'sts', 'data_path': 'sick.json'},

        #synthetic
        'synthetic_msmarco_halluFalse': {'task_type': 'bi_grounding', 'data_path': 'synthetic_msmarco_halluFalse.json'},
        'synthetic_msmarco_halluTrue': {'task_type': 'bi_grounding', 'data_path': 'synthetic_msmarco_halluTrue.json'},
    }

    ALL_TRAINING_MINICHECK_DATASETS = {
        'anli_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'anli_rbt_mnli_failed.json'},
        'anli_minicheck': {'task_type': 'bin_grounding', 'data_path': 'anli_minicheck.json'},
        
        'musique_full_train_minhop3': {'task_type': 'bin_grounding', 'data_path': 'musique_full_train_minhop3.json'},
        'musique_full_train_minhop3_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'musique_full_train_minhop3_rbt_mnli_failed.json'},
        
        'hotpot_qa_train_medium_multihop_gold': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_train_medium_multihop_gold.json'},
        'hotpot_qa_train_multihop_gold': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_train_multihop_gold.json'},
        'hotpot_qa_dev_multihop_gold': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_dev_multihop_gold.json'},
        'hotpot_qa_train_multihop_gold_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_train_multihop_gold_rbt_mnli_failed.json'},
        'hotpot_qa_dev_multihop_gold_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_dev_multihop_gold_rbt_mnli_failed.json'},
        
        'hotpot_qa_train_multihop_kg_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_train_multihop_kg_rbt_mnli_failed.json'},
        'hotpot_qa_dev_multihop_kg_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_dev_multihop_kg_rbt_mnli_failed.json'},
        

        'hotpot_qa_train_multihop_multidoc1': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_train_multihop_multidoc1.json'},
        'hotpot_qa_dev_multihop_multidoc1': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_dev_multihop_multidoc1.json'},
        
        'minicheck_c2d': {'task_type': 'bin_grounding', 'data_path': 'minicheck_c2d.json'},
        
        'hotpot_qa_dev_multihop_multidoc1_maxtoken150': {'task_type': 'bin_grounding', 'data_path': 'hotpot_qa_dev_multihop_multidoc1_maxtoken150.json'},
        
        'minicheck_d2c': {'task_type': 'bin_grounding', 'data_path': 'minicheck_d2c.json'},
        'd2c_v2': {'task_type': 'bin_grounding', 'data_path': 'd2c_v2.json'},
        'd2c_v2_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'd2c_v2_rbt_mnli_failed.json'},

        'd2c_v4_34hops_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'd2c_v4_34hops_rbt_mnli_failed.json'},
        'd2c_v4_2hops_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'd2c_v4_2hops_rbt_mnli_failed.json'},
        
        "d2c_v4_3hops": {'task_type': 'bin_grounding', 'data_path': 'd2c_v4_3hops.json'},
        "d2c_v4_4hops": {'task_type': 'bin_grounding', 'data_path': 'd2c_v4_4hops.json'},

        "d2c_v5_4hops": {'task_type': 'bin_grounding', 'data_path': 'd2c_v5_4hops.json'},
        "d2c_v5_3hops": {'task_type': 'bin_grounding', 'data_path': 'd2c_v5_3hops.json'},
        "d2c_v5_4hops_rbt_mnli_failed": {'task_type': 'bin_grounding', 'data_path': 'd2c_v5_4hops_rbt_mnli_failed.json'},
        "d2c_v5_3hops_rbt_mnli_failed": {'task_type': 'bin_grounding', 'data_path': 'd2c_v5_3hops_rbt_mnli_failed.json'},


        'AICopilot_tsv30%_inputlen1024_labeled_bi_label': {'task_type': 'bin_grounding', 'data_path': 'data_finetune_factcg/AICopilot_tsv30%_inputlen1024_labeled_bi_label.json'},
        'project_wednesday_tsv30%_inputlen1024_labeled_bi_label': {'task_type': 'bin_grounding', 'data_path': 'data_finetune_factcg/project_wednesday_tsv30%_inputlen1024_labeled_bi_label.json'},
        'bap_qa_tsv30%_inputlen1024_labeled_bi_label': {'task_type': 'bin_grounding', 'data_path': 'data_finetune_factcg/bap_qa_tsv30%_inputlen1024_labeled_bi_label.json'},
        'TestBase_tsv30%_inputlen1024_labeled_bi_label': {'task_type': 'bin_grounding', 'data_path': 'data_finetune_factcg/TestBase_tsv30%_inputlen1024_labeled_bi_label.json'},
        'NYC35_tsv30%_inputlen1024_labeled_bi_label': {'task_type': 'bin_grounding', 'data_path': 'data_finetune_factcg/NYC35_tsv30%_inputlen1024_labeled_bi_label.json'},
        'election_tsv30%_inputlen1024_labeled_bi_label': {'task_type': 'bin_grounding', 'data_path': 'data_finetune_factcg/election_tsv30%_inputlen1024_labeled_bi_label.json'},
        'Election_minicheckd2c': {'task_type': 'bin_grounding', 'data_path': 'Election_minicheckd2c.json'},
    }

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--accumulate-grad-batch', type=int, default=1)
    parser.add_argument('--num-epoch', type=int, default=3)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--warm-up-proportion', type=float, default=0.06)
    parser.add_argument('--adam-epsilon', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--val-check-interval', type=float, default=1. / 4)
    parser.add_argument('--devices', nargs='+', type=int, required=True)
    # TODO: set as avariables
    parser.add_argument('--model-name', type=str, default="microsoft/deberta-v3-large")
    parser.add_argument('--ckpt-save-path', type=str, required=True)
    parser.add_argument('--ckpt-comment', type=str, default="")
    parser.add_argument('--training-datasets', nargs='+', type=str, default=list(ALL_TRAINING_MINICHECK_DATASETS.keys()), choices=list(ALL_TRAINING_MINICHECK_DATASETS.keys()))
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--max-samples-per-dataset', type=int, default=500000)
    parser.add_argument('--do-mlm', type=bool, default=False)
    parser.add_argument('--ckpt-path', type=str, default="")
   
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    datasets = {
        name: {
            **ALL_TRAINING_MINICHECK_DATASETS[name],
            "size": args.max_samples_per_dataset,
            "data_path": os.path.join(args.data_path, ALL_TRAINING_MINICHECK_DATASETS[name]['data_path'])
        }
        for name in args.training_datasets
    }

    train(datasets, args)

