import torch
from grounding_model import GroundingModel
from transformers import AutoTokenizer, AutoConfig, DebertaV2ForSequenceClassification


device = torch.device("cuda")
model_name = "microsoft/deberta-v3-large"
pt_ligntning_model = GroundingModel.load_from_checkpoint(
    "./../ckpt/factcg_dbt.ckpt", model_name=model_name, strict=False).to(device)

config = AutoConfig.from_pretrained(model_name, num_labels=2,
                                    finetuning_task="text-classification", revision='main', token=None, cache_dir="./cache")
config.problem_type = "single_label_classification"
huggingface_model = DebertaV2ForSequenceClassification(
    AutoConfig.from_pretrained(model_name, config=config))
huggingface_tokenizer = AutoTokenizer.from_pretrained(model_name)

print("replace the layers.")
huggingface_model.deberta = pt_ligntning_model.base_model
huggingface_model.classifier = pt_ligntning_model.bin_layer
huggingface_model.pooler = pt_ligntning_model.pooler

print("upload to hugging face")
huggingface_tokenizer.push_to_hub("yaxili96/FactCG-DeBERTa-v3-Large")
huggingface_model.push_to_hub("yaxili96/FactCG-DeBERTa-v3-Large")
