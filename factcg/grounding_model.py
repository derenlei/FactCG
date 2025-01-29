import math
from typing import Optional, Tuple
from transformers import get_linear_schedule_with_warmup, AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from transformers import Adafactor
from transformers import BertModel, RobertaModel, PhiModel, PhiPreTrainedModel, PhiForSequenceClassification
from transformers import DebertaV2ForSequenceClassification, DebertaV2ForMaskedLM
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from dataclasses import dataclass
import copy
import os
import torch

class GroundingModelForMultitaskLearning(pl.LightningModule):
    def __init__(self, model_name='microsoft/deberta-v3-large', *args, **kwargs) -> None:
        super().__init__()
        # Already defined in lightning: self.device
        self.save_hyperparameters()
        self.validation_step_outputs = []

        self.model_name = model_name

        self.accumulate_loss = 0
        
        if 'deberta' in model_name:
            self.base_model = AutoModel.from_pretrained(model_name)
            self.pooler = DebertaV2ForSequenceClassification(AutoConfig.from_pretrained(model_name)).pooler
        elif 't5' in model_name.lower():
            self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif 'roberta' in model_name:
            self.base_model = RobertaModel(AutoConfig.from_pretrained(model_name))
    
        self.bin_layer = nn.Linear(self.base_model.config.hidden_size, 2)
        self.dropout = nn.Dropout(p=0.1)

    def half(self):
        if 'deberta' in self.model_name:
            self.pooler.half()
        self.base_model.half()
        self.bin_layer.half()

    def transfer_learning_init(self):
        # Experiment Note: freezing backbone model for transfer learning style finetuning didn't lead to promising result
        # self.base_model.requires_grad_(False)
        # self.pooler.requires_grad_(False)
        self.bin_layer.requires_grad_(False)

    def forward(self, batch):
        # # remove padding tokens per batch to pad to longest sequence
        max_length = torch.max(torch.sum(batch['attention_mask'], dim=1))
        batch['input_ids'] = batch['input_ids'][:, :max_length].squeeze(1)
        batch['attention_mask'] = batch['attention_mask'][:, :max_length].squeeze(1)

        if 'deberta' in self.model_name:
            base_model_output = self.base_model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch.keys() else None,
                # output_attentions=True
            )
            pooler_output = self.pooler(base_model_output.last_hidden_state)
            bi_label_score = self.bin_layer(self.dropout(pooler_output))
        # support binary for now
        elif 't5' in self.model_name.lower():
            # print("start!!!!!!!!!!!!!!!!!")
            
            batch['input_ids'] = batch['input_ids'].squeeze(1)
            batch['attention_mask'] = batch['attention_mask'].squeeze(1)
            decoder_input_ids = torch.zeros((batch['input_ids'].size(0),1), dtype=torch.long).to(batch['input_ids'].device)
            base_model_output = self.base_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], decoder_input_ids=decoder_input_ids)

            logits = base_model_output.logits.squeeze(1)
            label_logits = logits[:, torch.tensor([465, 2163])]
            bi_label_score = label_logits
        else:
            base_model_output = self.base_model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch.keys() else None,
            )
            bi_label_score = self.bin_layer(self.dropout(base_model_output.pooler_output)) ## pooled output for classification
        
        all_loss, loss_nums = [], []
        if 'mlm_label' in batch.keys(): ### 'mlm_label' and 'align_label' when training
            ce_loss_func = nn.CrossEntropyLoss(reduction='sum')
            bi_label_loss = ce_loss_func(bi_label_score.view(-1, 2), batch['align_label'].view(-1)) / math.log(2)
            bi_label_loss_num = torch.sum(batch['align_label'].view(-1) != -100)
            all_loss = [bi_label_loss]
            loss_nums = [bi_label_loss_num]

        return ModelOutput(
            all_loss=all_loss if 'mlm_label' in batch.keys() else None,
            loss_nums=loss_nums if 'mlm_label' in batch.keys() else None,
            bi_label_logits=bi_label_score,
            hidden_states=base_model_output.hidden_states if "t5" not in self.model_name.lower() else None,
            attentions=base_model_output.attentions if "t5" not in self.model_name.lower() else None
        )
            
    def training_step(self, train_batch, batch_idx):
        output = self(train_batch)
        losses = output.all_loss
        loss_nums = output.loss_nums
        assert len(loss_nums) == len(losses), 'loss_num should be the same length as losses'

        loss_bin_num = torch.sum(loss_nums[0])
        loss_bin = torch.sum(losses[0]) / loss_bin_num if loss_bin_num > 0 else 0.

        self.accumulate_loss += loss_bin

        self.log('train_loss', loss_bin, prog_bar=True)

        if batch_idx % 4 == 0:
            avg_loss = self.accumulate_loss / 4
            self.log('avg_train_loss', avg_loss, prog_bar=True)
            self.accumulate_loss = 0
        return loss_bin

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            # print(val_batch.keys)
            output = self(val_batch)

            losses = output.all_loss
            loss_nums = output.loss_nums
            assert len(loss_nums) == len(losses), 'loss_num should be the same length as losses'

            loss_bin_num = torch.sum(loss_nums[0])
            loss_bin = torch.sum(losses[0]) / loss_bin_num if loss_bin_num > 0 else 0.

            self.log('train_loss', loss_bin)
            self.validation_step_outputs.append(loss_bin)
        return loss_bin

    def validation_step_end(self, step_output):
        losses = step_output['losses']
        loss_nums = step_output['loss_nums']
        assert len(loss_nums) == len(losses), 'loss_num should be the same length as losses'

        loss_bin_num = torch.sum(loss_nums[0])
        loss_bin = torch.sum(losses[0]) / loss_bin_num if loss_bin_num > 0 else 0.

        self.log('train_loss', loss_bin)
        self.validation_step_outputs.append(loss_bin)
        return loss_bin

    def on_validation_epoch_end(self):
        total_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.warmup_steps_portion * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def on_load_checkpoint(self, checkpoint):
        unwanted_keys = [
            'tri_layer.weight',
            'tri_layer.bias',
            'reg_layer.weight',
            'reg_layer.bias',
            'bin_finetuned_layer.weight',
            'bin_finetuned_layer.bias'
        ]
        for key in unwanted_keys:  
            if key in checkpoint['state_dict']:  
                del checkpoint['state_dict'][key] 

@dataclass
class ModelOutput():
    all_loss: Optional[list] = None
    loss_nums: Optional[list] = None
    bi_label_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None