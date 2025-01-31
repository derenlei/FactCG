FACTCG_CKPT="ckpt/factcg_dbt.ckpt"
CUDA_VISIBLE_DEVICES=1 python3 benchmark.py --threshold-setting tune --factcg --factcg-model-name microsoft/deberta-v3-large --factcg-ckpt $FACTCG_CKPT