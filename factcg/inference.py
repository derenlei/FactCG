from logging import warning
from nltk.tokenize import sent_tokenize
import torch
from .grounding_model import GroundingModelForMultitaskLearning
from transformers import AutoConfig, AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
import os
from typing import List
import nltk
import numpy as np
import math

from nltk.tokenize.texttiling import TextTilingTokenizer

class Inferencer():
    def __init__(self, ckpt_path, model_name='microsoft/deberta-v3-large', batch_size=32, verbose=True) -> None:
        self.prompt = "Determine if the hypothesis is true given the premise?\n\nPremise: {Premise}\n\nHypothesis: {Hypothesis}\n\n[CLS]"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            warning('CUDA not available, loading all tensors into cpu')
        if ckpt_path is not None:
            self.model = GroundingModelForMultitaskLearning.load_from_checkpoint(ckpt_path, model_name=model_name, strict=False).to(self.device)
        else:
            warning('loading UNTRAINED model!')
            self.model = GroundingModelForMultitaskLearning(model_name=model_name).to(self.device)
        self.model_name = model_name
        self.model.eval()
        self.batch_size = batch_size
        self.tokenizer = self.model.tokenizer
        self.softmax = nn.Softmax(dim=-1)
        self.disable_progress_bar_in_inference = False
        self.verbose = verbose

        # emperically observed that w=20 leads to worse result on some measurement sets.
        #self.textiling = TextTilingTokenizer(w=10)

    def inference_example_real_batch(self, premise: list, hypo: list):
        """
        inference a example,
        premise: list
        hypo: list
        using self.inference to batch the process

        SummaC Style aggregation
        """
        self.disable_progress_bar_in_inference = False
        assert len(premise) == len(hypo), "Premise must has the same length with Hypothesis!"

        chunksize_per_example = []
        premise_sent_mat = []
        hypo_sents_mat = []

        for one_pre, one_hypo in zip(premise, hypo):
            one_pre_sents = self.chunking_src_mop(one_pre)
            # one_pre_sents = self.chunk_src_minicheck(one_pre)
            one_hypo_sents = [one_hypo] * len(one_pre_sents)

            premise_sent_mat.extend(one_pre_sents)
            hypo_sents_mat.extend(one_hypo_sents)
            chunksize_per_example.append((1, len(one_pre_sents)))

        assert len(hypo) == len(chunksize_per_example)
        assert type(premise_sent_mat[0]) is str
        assert type(hypo_sents_mat[0]) is str

        output_score_all_flat, attentions_all = self.inference(premise_sent_mat, hypo_sents_mat)

        output_score_all = []
        best_chunk_ids = []
        output_score_per_example = torch.split(output_score_all_flat, [math.prod(n) for n in chunksize_per_example])#, dim=1)
        for output_score, chunk_size in zip(output_score_per_example, chunksize_per_example):
            hypo_len, premise_len = chunk_size
            best_chunk_id = output_score.view(hypo_len, premise_len).max(dim=1).indices[0].item()
            output_score_val = output_score.view(hypo_len, premise_len).max(dim=1).values.mean().item() ### sum or mean depends on the task/aspect
            output_score_all.append(output_score_val)
            best_chunk_ids.append(best_chunk_id)
        return output_score_all, best_chunk_ids

    def chunk_src_minicheck(self, src, max_chunk_size=500):
        def sent_tokenize_with_newlines(text):
            blocks = text.split('\n')
            
            tokenized_blocks = [sent_tokenize(block) for block in blocks]
            tokenized_text = []
            for block in tokenized_blocks:
                tokenized_text.extend(block)
                tokenized_text.append('\n')  

            return tokenized_text[:-1]  

        def chunks(lst, n):
            """Yield successive chunks from lst with each having approximately n tokens.

            For flan-t5, we split using the white space;
            For roberta and deberta, we split using the tokenization.
            """
            if "t5" in self.model_name:
                current_chunk = []
                current_word_count = 0
                for sentence in lst:
                    sentence_word_count = len(sentence.split())
                    if current_word_count + sentence_word_count > n:
                        yield ' '.join(current_chunk)
                        current_chunk = [sentence]
                        current_word_count = sentence_word_count
                    else:
                        current_chunk.append(sentence)
                        current_word_count += sentence_word_count
                if current_chunk:
                    yield ' '.join(current_chunk)
            else:
                current_chunk = []
                current_token_count = 0
                for sentence in lst:
                    sentence_word_count = len(self.tokenizer(
                        sentence, padding=False, add_special_tokens=False, 
                        max_length=self.max_model_len, truncation=True)['input_ids'])
                    if current_token_count + sentence_word_count > n:
                        yield ' '.join(current_chunk)
                        current_chunk = [sentence]
                        current_token_count = sentence_word_count
                    else:
                        current_chunk.append(sentence)
                        current_token_count += sentence_word_count
                if current_chunk:
                    yield ' '.join(current_chunk)

        src_sents = sent_tokenize_with_newlines(src)
        src_sents = src_sents or ['']

        src_chunks = [chunk.replace(" \n ", '\n').strip() for chunk in chunks(src_sents, max_chunk_size)]
        src_chunks = [chunk for chunk in src_chunks if chunk != '']
        return src_chunks

    # todo: duplicated function, abstract it out to avoid two copies.
    def chunking_src_mop(self, src : str, max_chunk_size : int = 550): #350
        def get_tokens(text:str) -> List[str]:
            return nltk.word_tokenize(text)

        def count_tokens(text : str) -> int:
            # if "deberta" in self.model_name:
            #     return len(self.tokenizer(
            #             text, padding=False, add_special_tokens=False, 
            #             max_length=2048, truncation=True)['input_ids'])
            # else:
            return len(get_tokens(text))
        def split_into_sentences(text : str, offset : int = 0) -> List[str]:
            sentences : List[str] = sent_tokenize(text)
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
        chunks : List[str] = []
        chunk_size : int = 0
        chunk : str = ""
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

    def chunking_src_mop_sliding_window(self, src : str, max_chunk_size : int = 400, max_step_size : int = 200):
        def get_tokens(text:str) -> List[str]:
            return nltk.word_tokenize(text)

        def count_tokens(text : str) -> int:
            return len(get_tokens(text))
        def split_into_sentences(text : str, offset : int = 0) -> List[str]:
            sentences : List[str] = sent_tokenize(text)
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
        
        def slide_window(sentences: List[str], max_step_size : int) -> List[str]:
            step_size : int = 0
            for i in range(len(sentences)):
                s = sentences[i]
                token_count = count_tokens(s["text"])
                if step_size + token_count <= max_step_size:
                    step_size += token_count
                else:
                    break
            return sentences[i+1:]
        
        def get_chunk(sentence : List[str], max_chunk_size : int) -> List[str]:
            chunk_size : int = 0
            chunk : str = ""
            for s in sentences:
                token_count = count_tokens(s["text"])
                if chunk_size + token_count <= max_chunk_size:
                    chunk = "\n".join([chunk, s["text"]]).strip('\n')
                    chunk_size += token_count
                else:
                    break
            return chunk.strip('\n')
            
        sentences = split_into_sentences(src)
        chunks : List[str] = []
        while len(sentences) > 0:
            chunk = get_chunk(sentences, max_chunk_size = max_chunk_size)
            chunks.append(chunk)
            sentences = slide_window(sentences, max_step_size = max_step_size)
        return chunks
    
    def chunking_src_textiling(self, src : str, max_chunk_size : int = 400):
        def get_tokens(text:str) -> List[str]:
            return nltk.word_tokenize(text)
        def count_tokens(text : str) -> int:
            return len(get_tokens(text))

        try:
            chunks : List[str] = []
            if "\n\n" not in src and "\n" in src:
                src = src.replace("\n", "\n\n")

            chunks = self.textiling.tokenize(src)
            chunks = [chunk.strip() for chunk in chunks]
            for chunk in chunks:
                if count_tokens(chunk) > 1500:
                    return self.chunking_src_mop(src)
                else:
                    continue
        except:
            # print(src)
            # chunks = [src]
            return self.chunking_src_mop(src)
        return chunks
    
    def np_softmax(self, x):
        r=np.exp(x - np.max(x))
        return r/r.sum(axis=0)

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
        output_attentions = []
        for mini_batch in tqdm(batch, desc="Evaluating", disable=not self.verbose or self.disable_progress_bar_in_inference):
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(mini_batch)
                model_output_bin = model_output.bi_label_logits # Temperature Scaling / 2.5
                if model_output.attentions:
                    model_output.attentions = list(model_output.attentions)
                    for i in range(len(model_output.attentions)):
                        model_output.attentions[i] = model_output.attentions[i].detach().cpu()
                    output_attentions.append(model_output.attentions)
                
                model_output_bin = self.softmax(model_output_bin).cpu()
                output_score_bin.append(model_output_bin[:,1])

        if output_score_bin:
            output_score_bin = torch.cat(output_score_bin)

        return output_score_bin, output_attentions
    
    def batch_tokenize(self, premise, hypo):
        """
        input premise and hypos are lists
        """
        assert isinstance(premise, list) and isinstance(hypo, list)
        assert len(premise) == len(hypo), "premise and hypo should be in the same length."

        batch = []
        for mini_batch_pre, mini_batch_hypo in zip(self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)):
            mini_batch = self.tokenize(mini_batch_pre, mini_batch_hypo)
            batch.append(mini_batch)

        return batch

    def tokenize(self, premise_list, hypo_list):
        if "phi-cls" in self.model_name:
            merged_input_list = []
            for premise, hypo in zip(premise_list, hypo_list):
                # merged_input_list.append(f"{premise} [SEP] {hypo}")
                merged_input = self.prompt.format(**{"Premise":premise, "Hypothesis":hypo})
                merged_input_list.append(merged_input)
            
            try:
                output = self.tokenizer(merged_input_list, truncation='only_first', padding='longest', max_length=2048, return_tensors='pt')
            except:
                warning('text_b too long...')
                output = self.tokenizer(merged_input_list, truncation=True, padding='longest', max_length=2048, return_tensors='pt')
        elif "t5" or "phi2" in self.model_name.lower():
            # text_list = [self.tokenizer.eos_token.join([one_doc, one_claim]) for one_doc, one_claim in zip(premise_list, hypo_list)]
            # prompt = "Read the following paragraph and determine if the hypothesis is true:\n\n{text_a}\n\nHypothesis: {text_b}n\nOPTIONS:\n- true\n- false"
            
            prompt = "{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that \"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is "
            # prompt = "{text_a}\n\nBased on that paragraph can we conclude that this sentence is true?\n{text_b}\n\nOPTIONS:\n- Yes\n- No\n"
            # prompt = "{text_a}\n\nCan we draw the following conclusion?\n{text_b}\n\nOPTIONS:\n- yes\n- no"
            
            text_list = [prompt.format(text_a=one_doc, text_b=one_claim) for one_doc, one_claim in zip(premise_list, hypo_list)]
            # text_list = [f"{one_doc} [SEP] {one_claim}" for one_doc, one_claim in zip(premise_list, hypo_list)]
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
                output = self.tokenizer(premise_list, hypo_list, truncation='only_first', padding='longest', max_length=2048, return_tensors='pt')
            except:
                warning('text_b too long...')
                output = self.tokenizer(premise_list, hypo_list, truncation=True, padding='longest', max_length=2048, return_tensors='pt')
        
        return output

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def eval(self, premise, hypo):
        with torch.no_grad():
            out_score, best_chunk_ids =  self.inference_example_real_batch(premise, hypo)
            return torch.tensor(out_score), best_chunk_ids
