CUDA_VISIBLE_DEVICES=1,2,3 python3 train.py \
--seed 2024 \
--batch-size 2 \
--accumulate-grad-batch 8 \
--num-epoch 1 \
--devices 0 1 2 \
--training-datasets anli_minicheck CG2C_hotpot_qa_rbt_mnli_failed CG2C_musique_minhop3_rbt_mnli_failed minicheck_c2d \
--model-name microsoft/deberta-v3-large \
--ckpt-save-path ckpt_factcg_dbt_stage1/ \
--data-path data/training_stage1/ \
--max-samples-per-dataset 500000

CUDA_VISIBLE_DEVICES=1,2,3 python3 train.py \
--seed 2024 \
--batch-size 2 \
--accumulate-grad-batch 8 \
--num-epoch 1 \
--devices 0 1 2 \
--training-datasets CG2C_doc minicheck_d2c \
--model-name microsoft/deberta-v3-large \
--ckpt-path ./ckpt_factcg_dbt_stage1/microsoft-deberta-v3-large_final.ckpt \
--ckpt-save-path ckpt_factcg_dbt_stage2/ \
--data-path data/training_stage2/ \
--max-samples-per-dataset 500000