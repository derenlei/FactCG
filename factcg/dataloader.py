import json
import logging
import random
from typing import Optional, Sized, List, Dict
import numpy as np
from .utils import INSTRUCTION_TEMPLATE
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from torch.utils.data import Dataset, Sampler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AlignmentDataset(Dataset):
    def __init__(
            self,
            dataset : List[Dict], 
            tokenizer : AutoTokenizer, 
            model_name : str ='microsoft/deberta-v3-large', 
            tokenizer_max_length=2048,
        ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.dataset_type_dict = dict()

        self.dataset = dataset
        self.dataset_type_dict_init()

    def encode(self, text_a, text_b, options):
        prompt = f"{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that \"{text_b}\"?\n\nOPTIONS:\n- Yes\n-No\nI think the answer is "
        try:
            output = self.tokenizer(
                prompt, 
                truncation='only_first', 
                padding='max_length', 
                return_tensors="pt", 
                max_length=self.tokenizer_max_length
            ) 
        except:
            logging.warning('text_b too long...')
            output = self.tokenizer(
                prompt, 
                truncation=True, 
                padding='max_length', 
                return_tensors="pt", 
                max_length=self.tokenizer_max_length
            )
        return output

    def dataset_type_dict_init(self):
        for i, item in enumerate(self.dataset):
            try:
                self.dataset_type_dict[item['task']].append(i)
            except:
                self.dataset_type_dict[item['task']] = [i]
                
    def process_bin_grounding(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][0]
        label = self.dataset[index]['orig_label']

        tokenized_pair = self.encode(text_a, text_b, options=["Yes, No"])
        return (
            torch.tensor(tokenized_pair['input_ids']), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(label), # binary
        )

    def get_task_processor(self, task):
        if task == 'bin_grounding':
            return self.process_bin_grounding
        else:
            assert False, f"Task {task} not supported."

    def __getitem__(self, index):
        task = self.dataset[index]['task']
        task_processor = self.get_task_processor(task)
        input_ids, attention_mask, token_type_ids, align_label = task_processor(index)
 
        if token_type_ids is not None:
            print("HERE!!!!!!!!!!!!!!!!!!!!!")
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'align_label': align_label,
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'align_label': align_label,
            }

    def __len__(self):
        return len(self.dataset)


class PropSampler(Sampler[int]):
    def __init__(self, data_source: Optional[Sized]) -> None:
        super().__init__(data_source)
        self.K = 500000
        print("Initializing Prop Sampler")

        self.data_positions = dict()
        for i, example in tqdm(enumerate(data_source), desc="Initializing Sampler"):
            if example['dataset_name'] in self.data_positions.keys():
                self.data_positions[example['dataset_name']].append(i)
            else:
                self.data_positions[example['dataset_name']] = [i]
        self.all_dataset_names = list(self.data_positions.keys())
        self.dataset_lengths = {each:len(self.data_positions[each]) for each in self.data_positions}

        self.dataset_props = {each: min(self.dataset_lengths[each], self.K) for each in self.dataset_lengths}
        self.dataset_props_sum = sum([self.dataset_props[each] for each in self.dataset_props])
        
        print("Finish Prop Sampler initialization.")

    def __iter__(self):
        iter_list = []
        for each in self.dataset_props:
            iter_list.extend(np.random.choice(
                self.data_positions[each], size=self.dataset_props[each], replace=False).tolist())

        random.shuffle(iter_list)

        yield from iter_list

    def __len__(self):
        return self.dataset_props_sum


class AlignmentDataLoader(LightningDataModule):
    def __init__(self,dataset_config, val_dataset_config=None, sample_mode='seq', model_name='bert-base-uncased', is_finetune=False, tokenizer_max_length=512, train_batch_size=32, eval_batch_size=4, num_workers=16, train_eval_split=0.8, **kwargs):
        super().__init__(**kwargs)
        assert sample_mode in ['seq', 'proportion']
        self.sample_mode = sample_mode
        self.dataset_config = dataset_config
        self.val_dataset_config = val_dataset_config
        self.num_workers = num_workers
        self.train_eval_split = train_eval_split
        self.tokenizer_max_length = tokenizer_max_length
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        self.train_bach_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            print("Already Initilized LightningDataModule!")
            return

        self.init_training_set()

        self.dataset = dict()
        self.dataset['train'] = AlignmentDataset(dataset=self.raw_dataset[:int(self.train_eval_split*len(self.raw_dataset))], tokenizer=self.tokenizer, model_name=self.model_name)
        self.dataset['test'] = AlignmentDataset(dataset=self.raw_dataset[int(self.train_eval_split*len(self.raw_dataset)):], tokenizer=self.tokenizer, model_name=self.model_name)

    
    def init_training_set(self):
        self.raw_dataset = []
        if self.sample_mode == 'seq':
            for each_dataset in self.dataset_config:
                dataset_length = sum([1 for line in open(
                    self.dataset_config[each_dataset]['data_path'], 'r', encoding='utf8')])
                dataset_length_limit = self.dataset_config[each_dataset]['size'] if isinstance(
                    self.dataset_config[each_dataset]['size'], int) else int(self.dataset_config[each_dataset]['size'] * dataset_length)
                with open(self.dataset_config[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                    try:
                        for i, example in enumerate(f):
                            if i >= dataset_length_limit:
                                break
                            self.raw_dataset.append(
                                json.loads(example))  # + dataset_name
                    except:
                        print(example)
                        data = json.loads(example)
                        print(f"failed to load data from {each_dataset}.json, exiting...")
                        exit()

            random.shuffle(self.raw_dataset)

        elif self.sample_mode == 'proportion':
            for each_dataset in tqdm(self.dataset_config, desc="Loading data from disk..."):
                with open(self.dataset_config[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                    try:
                        for i, example in enumerate(f):
                            jsonobj = json.loads(example)
                            jsonobj['dataset_name'] = each_dataset
                            self.raw_dataset.append(jsonobj)  # + dataset_name
                    except:
                        print(f"failed to load data from {each_dataset}.json, exiting...")
                        exit()

            random.shuffle(self.raw_dataset)

    def prepare_data(self) -> None:
        AutoTokenizer.from_pretrained(self.model_name)

    def train_dataloader(self):
        if self.sample_mode == 'seq':
            return DataLoader(self.dataset['train'], batch_size=self.train_bach_size, shuffle=True, num_workers=self.num_workers)
        elif self.sample_mode == 'proportion':
            return DataLoader(self.dataset['train'], batch_size=self.train_bach_size, sampler=PropSampler(self.raw_dataset[:int(self.train_eval_split*len(self.raw_dataset))]), num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)
