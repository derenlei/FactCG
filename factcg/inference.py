from logging import warning
from nltk.tokenize import sent_tokenize
import torch
from .utils import INSTRUCTION_TEMPLATE
from .grounding_model import GroundingModel
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from tqdm import tqdm
from typing import List
import nltk
import math


class Inferencer():
    def __init__(self, ckpt_path=None, model_name='microsoft/deberta-v3-large', batch_size=32, verbose=True, use_hf_ckpt=True) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            warning('CUDA not available, loading all tensors into cpu')
        if use_hf_ckpt:
            if not 'deberta' in model_name.lower():
                raise ValueError("Only DeBERTa-v3-Large has model checkpoint on huggingface")
            ckpt_path = "yaxili96/FactCG-DeBERTa-v3-Large"
            config = AutoConfig.from_pretrained(
                ckpt_path, num_labels=2, finetuning_task="text-classification", revision='main', token=None, cache_dir="./cache")
            config.problem_type = "single_label_classification"

            self.tokenizer = AutoTokenizer.from_pretrained(
                ckpt_path, use_fast=True, revision='main', token=None, cache_dir="./cache")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                ckpt_path, config=config, revision='main', token=None, ignore_mismatched_sizes=False, cache_dir="./cache").to(self.device)
        elif ckpt_path is not None:
            self.model = GroundingModel.load_from_checkpoint(
                ckpt_path, model_name=model_name, strict=False).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            warning('loading UNTRAINED model!')
            self.model = GroundingModel(
                model_name=model_name).to(self.device)
        self.use_hf_ckpt = use_hf_ckpt
        self.model_name = model_name
        self.model.eval()
        self.batch_size = batch_size
        self.softmax = nn.Softmax(dim=-1)
        self.disable_progress_bar_in_inference = False
        self.verbose = verbose

    def batch_inference(self, premise: list, hypo: list):
        """
        inference a example,
        premise: list
        hypo: list
        using self.inference to batch the process

        SummaC Style aggregation
        """
        self.disable_progress_bar_in_inference = False
        assert len(premise) == len(
            hypo), "Premise must has the same length with Hypothesis!"

        chunksize_per_example = []
        premise_sent_mat = []
        hypo_sents_mat = []

        for one_pre, one_hypo in zip(premise, hypo):
            one_pre_sents = self.chunking_src(one_pre)
            one_hypo_sents = [one_hypo] * len(one_pre_sents)

            premise_sent_mat.extend(one_pre_sents)
            hypo_sents_mat.extend(one_hypo_sents)
            chunksize_per_example.append((1, len(one_pre_sents)))

        assert len(hypo) == len(chunksize_per_example)
        assert type(premise_sent_mat[0]) is str
        assert type(hypo_sents_mat[0]) is str

        output_score_all_flat = self.inference(
            premise_sent_mat, hypo_sents_mat)

        output_score_all = []
        best_chunk_ids = []
        output_score_per_example = torch.split(
            output_score_all_flat, [math.prod(n) for n in chunksize_per_example])
        for output_score, chunk_size in zip(output_score_per_example, chunksize_per_example):
            hypo_len, premise_len = chunk_size
            best_chunk_id = output_score.view(
                hypo_len, premise_len).max(dim=1).indices[0].item()
            output_score_val = output_score.view(
                hypo_len, premise_len).max(dim=1).values.mean().item()
            output_score_all.append(output_score_val)
            best_chunk_ids.append(best_chunk_id)
        return output_score_all, best_chunk_ids

    def chunking_src(self, src: str, max_chunk_size: int = 550):
        def get_tokens(text: str) -> List[str]:
            return nltk.word_tokenize(text)

        def count_tokens(text: str) -> int:
            return len(get_tokens(text))

        def split_into_sentences(text: str, offset: int = 0) -> List[str]:
            sentences: List[str] = sent_tokenize(text)
            ret = []
            idx = offset
            len_of_text = len(text)
            for sentence in sentences:
                # increment idx until we match the character at idx to the first character of the sentence
                first_char_of_sentence = sentence[0]
                while idx < len_of_text and text[idx] != first_char_of_sentence:
                    idx += 1
                start = idx
                end = idx + len(sentence)
                tmp = {
                    "text": sentence,
                    "start": start,
                    "end": end
                }
                idx = end + 1
                ret.append(tmp)
            return ret
        sentences = split_into_sentences(src)
        chunks: List[str] = []
        chunk_size: int = 0
        chunk: str = ""
        for s in sentences:
            token_count = count_tokens(s["text"])
            if chunk_size + token_count <= max_chunk_size:
                chunk = "\n".join([chunk, s["text"]]).strip('\n')
                chunk_size += token_count
            else:
                chunks.append(chunk.strip('\n'))
                chunk = s["text"]
                chunk_size = token_count

        if chunk_size > 0:
            chunks.append(chunk.replace(" \n ", '\n').strip('\n').strip())

        return chunks

    def inference(self, premise, hypo):
        """
        inference a list of premise and hypo

        Standard aggregation
        """
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]

        batch = self.batch_tokenize(premise, hypo)

        output_score_bin = []
        for mini_batch in tqdm(batch, desc="Evaluating", disable=not self.verbose or self.disable_progress_bar_in_inference):
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                if self.use_hf_ckpt and 'deberta' in self.model_name.lower():
                    # hugging face model inference
                    mini_batch = {k: v.to(self.model.device)
                                  for k, v in mini_batch.items()}
                    model_output = self.model(**mini_batch)
                    model_output_bin = model_output.logits
                else:
                    model_output = self.model(mini_batch)
                    model_output_bin = model_output.bi_label_logits  # Temperature Scaling / 2.5
                model_output_bin = self.softmax(model_output_bin).cpu()
                output_score_bin.append(model_output_bin[:, 1])

        if output_score_bin:
            output_score_bin = torch.cat(output_score_bin)

        return output_score_bin

    def batch_tokenize(self, premise, hypo):
        """
        input premise and hypos are lists
        """
        assert isinstance(premise, list) and isinstance(hypo, list)
        assert len(premise) == len(
            hypo), "premise and hypo should be in the same length."

        batch = []
        for mini_batch_pre, mini_batch_hypo in zip(self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)):
            mini_batch = self.tokenize(mini_batch_pre, mini_batch_hypo)
            batch.append(mini_batch)

        return batch

    def tokenize(self, premise_list, hypo_list):
        if "t5" in self.model_name.lower() or "deberta" in self.model_name.lower():
            text_list = [INSTRUCTION_TEMPLATE.format(
                text_a=one_doc, text_b=one_claim) for one_doc, one_claim in zip(premise_list, hypo_list)]
            self.tokenizer.pad_token = self.tokenizer.eos_token
            try:
                output = self.tokenizer(
                    text_list,
                    max_length=2048,
                    truncation='only_first',
                    padding='longest',
                    return_tensors="pt"
                )
            except:
                logging.warning('text_b too long...')
                output = self.tokenizer(
                    text_list,
                    truncation=True,
                    padding='longest',
                    return_tensors="pt",
                    max_length=2048
                )
        else:
            try:
                output = self.tokenizer(
                    premise_list,
                    hypo_list,
                    truncation='only_first',
                    padding='longest',
                    max_length=2048,
                    return_tensors='pt'
                )
            except:
                warning('text_b too long...')
                output = self.tokenizer(
                    premise_list,
                    hypo_list,
                    truncation=True,
                    padding='longest',
                    max_length=2048,
                    return_tensors='pt'
                )

        return output

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def eval(self, premise, hypo):
        with torch.no_grad():
            out_score, best_chunk_ids = self.batch_inference(premise, hypo)
            return torch.tensor(out_score), best_chunk_ids
