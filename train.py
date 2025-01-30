from pytorch_lightning import Trainer, seed_everything
from factcg.dataloader import AlignmentDataLoader
from factcg.grounding_model import GroundingModelForMultitaskLearning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
import os
import torch
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

    logger = TensorBoardLogger(
        "logs", name=args.ckpt_save_path.replace("/", "").replace(".", ""))
    trainer = Trainer(
        accelerator = 'gpu', 
        max_epochs = args.num_epoch, 
        devices = args.devices, 
        strategy = "ddp_find_unused_parameters_true", 
        precision = precision,
        callbacks = [checkpoint_callback],
        accumulate_grad_batches = args.accumulate_grad_batch,
        logger = logger
    )
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(os.path.join(
        args.ckpt_save_path, f"{checkpoint_name}_final.ckpt"))
    print("Training is finished.")


if __name__ == "__main__":
    TRAINING_DATASETS = {
        # Stage 1 training
        'anli_minicheck': {'task_type': 'bin_grounding', 'data_path': 'anli_minicheck.json'},
        'CG2C_hotpot_qa_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'CG2C_hotpot_qa_rbt_mnli_failed.json'},
        'CG2C_musique_minhop3_rbt_mnli_failed': {'task_type': 'bin_grounding', 'data_path': 'CG2C_musique_minhop3_rbt_mnli_failed.json'},
        'minicheck_c2d': {'task_type': 'bin_grounding', 'data_path': 'minicheck_c2d.json'},
        
        # Stage 2 training
        'CG2C_doc': {'task_type': 'bin_grounding', 'data_path': 'CG2C_doc.json'},
        'minicheck_d2c': {'task_type': 'bin_grounding', 'data_path': 'minicheck_d2c.json'},
    }

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
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
    parser.add_argument('--model-name', type=str, default="microsoft/deberta-v3-large")
    parser.add_argument('--ckpt-save-path', type=str, required=True)
    parser.add_argument('--ckpt-comment', type=str, default="")
    parser.add_argument('--training-datasets', nargs='+', type=str, default=list(TRAINING_DATASETS.keys()), choices=list(TRAINING_DATASETS.keys()))
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--max-samples-per-dataset', type=int, default=500000)
    parser.add_argument('--ckpt-path', type=str, default="")

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    datasets = {
        name: {
            **TRAINING_DATASETS[name],
            "size": args.max_samples_per_dataset,
            "data_path": os.path.join(args.data_path, TRAINING_DATASETS[name]['data_path'])
        }
        for name in args.training_datasets
    }

    train(datasets, args)
