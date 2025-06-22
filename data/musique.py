from datasets import load_dataset
import json
from gpt_caller import GPTCaller
from tqdm import tqdm
import random
from minicheck.minicheck import MiniCheck
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
import random

random.seed(2024)
# Dataset({
#     features: ['id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable'],
#     num_rows: 19938
# }

def get_declaratve_sentence(multihop_dataset, gpt_caller):
    for data in tqdm(multihop_dataset):
        # data["source"] = []
        # for paragraph in data["paragraphs"]:
        #     if paragraph["is_supporting"]:
        #         data["source"].append(paragraph)
        question = data["question"]
        answer = data["answer"]
        paylaod = {'prompt': gpt_caller.create_qa2d_prompt(question, answer)}
        declarative_sent = gpt_caller.process_payload(paylaod)["output"].strip()
        data["declarative_sentence"] = declarative_sent
    # Save dataset as new json file
    output_file_path = '/home/derenlei/preprocessing/musique_train_minhop3_declarative.json'
    with open(output_file_path, 'w') as output_file:
        json.dump(multihop_dataset, output_file)

def extract_triple(multihop_dataset, gpt_caller):
    with open('/home/derenlei/data_synthetic/musique_minhop3_relevant_triples.json', 'w') as fout:
        for data in tqdm(multihop_dataset):
            try:
                hypothesis = data["declarative_sentence"]
                facts = ""
                for question in data["question_decomposition"]:
                    sent_id = question["paragraph_support_idx"]
                    fact = data["paragraphs"][sent_id]["paragraph_text"]
                    facts += f"{fact}\n"
                facts = facts.strip()
                # print("hypothesis", hypothesis)

                paylaod = {'prompt': gpt_caller.create_content_graph_prompt(facts)}
                content_graph_string = gpt_caller.process_payload(paylaod)["output"]
                # print("content_graph",content_graph_string)
                content_graph = [tuple(triple.split(', ')) for triple in content_graph_string.split('\n')]
                data["content_graph"] = content_graph
                # print("content_graph",content_graph)
                
                paylaod = {'prompt': gpt_caller.create_relevant_triple_prompt(content_graph_string, hypothesis)}
                relevant_triple_string = gpt_caller.process_payload(paylaod)["output"].replace("\n\n", "\n")
                # print("relevant_triple_string",relevant_triple_string)
                relevant_triples_raw = [tuple(triple.split(', ')) for triple in relevant_triple_string.split('\n')]

                # filter gpt output prefixes
                relevant_triples = []
                for triple in relevant_triples_raw:
                    if len(triple) >= 3:
                        relevant_triples.append(triple)

                data["relevant_triples"] = relevant_triples            
                # print("relevant_triples", relevant_triples)

                json.dump(data, fout)
                fout.write('\n')
            except:
                print("error when calling gpt, save raw data")
                json.dump(data, fout)
                fout.write('\n')

def get_gold_grounding_data_rm_entity(multihop_dataset, gpt_caller):
    with open('/home/derenlei/data_synthetic/musique_minhop3_kg_noisy_facts.json', 'w') as fout:
        for data in tqdm(multihop_dataset):
            # if "relevant_triples" not in data or "corrupted_fact" in data:
            #     continue
            hypothesis = data["declarative_sentence"]
            relevant_triples = data["relevant_triples"]

            facts, facts_ids = [], []
            for sentence in data["paragraphs"]:
                if sentence["is_supporting"]:
                    fact = sentence["paragraph_text"]
                    facts.append(fact)
                    facts_ids.append(sentence["idx"])

            # # join facts as below for prompt:
            # # -fact1
            # # -fact2
            facts_sentences = "\n-".join(facts)
            facts_sentences = f"-{facts_sentences}"

            # # randomly select 1 triple to remove its edge (relation)
            
            # # skip falsely parsed gpt output heading sentences
            relevant_triples = [triple for triple in relevant_triples if len(triple) >= 3]
            if len(relevant_triples) == 0:
                print("no relevant triples, save raw data")
                json.dump(data, fout)
                fout.write('\n')
                continue
            triple_to_remove = random.choice(relevant_triples)

            # # e2 has more than 1 word
            # # e.g. ['(Level 3 Communications', 'headquartered in', 'Broomfield', 'Colorado)']
            if len(triple_to_remove) > 3:
                for i in range(3, len(triple_to_remove)):
                    triple_to_remove[2] += " " + triple_to_remove[i]
                    print("triple have more than 3 items, parsed triple_to_remove", triple_to_remove)

            entities = triple_to_remove[0].strip("(") + ", " + triple_to_remove[2].strip(")")

            try:
                paylaod = {'prompt': gpt_caller.create_relation_removal_prompt(facts_sentences, entities)}
                corrupted_fact_sentence = gpt_caller.process_payload(paylaod)["output"]
                # parse gpt output with bullet points to list of facts
                corrupted_fact = corrupted_fact_sentence.split("\n-")
                corrupted_fact = [fact.strip("-").strip() for fact in corrupted_fact]
                assert len(corrupted_fact) == len(facts), f"{len(corrupted_fact)} != {len(facts)}"
                data["corrupted_fact"] = corrupted_fact
            except:
                print("error when calling gpt, save raw data")
            json.dump(data, fout)
            fout.write('\n')


# return {task, text_a, text_b=[],text_c=[],orig_label} line of json
def get_training_data(multihop_dataset):
    with open('/home/derenlei/data_synthetic/musique_full_train_minhop3.json', 'w') as fout:
        for data in tqdm(multihop_dataset[:3]):
            data["source"] = []
            for paragraph in data["paragraphs"]:
                if paragraph["is_supporting"] and data["answerable"] ==0:
                    data["source"].append(paragraph["paragraph_text"])
            random.shuffle(data["source"])
            cur_data = {
                "task": "bin_grounding",
                "text_a": "\n".join(data["source"]),
                "text_b": [data["declarative_sentence"]],
                "text_c": [],
                "orig_label": 1 if data["answerable"] else 0
            }
            json.dump(cur_data, fout)
            fout.write('\n')

def filter_hard_data_by_model(multihop_dataset):
    max_length = 512
    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    #     # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contradiction"
    # },
    # hg_model_hub_name = "FacebookAI/roberta-large-mnli" 
    #  "id2label": {
    #     "0": "CONTRADICTION",
    #     "1": "NEUTRAL",
    #     "2": "ENTAILMENT"
    #   },

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    config = AutoConfig.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # with open('/home/derenlei/data_synthetic/musique_minhop3_kg_noisy_facts_posneg_rbt_mnli_failed.json', 'w') as fout:
    with open('/home/derenlei/data_synthetic/musique_full_train_minhop3_rbt_anli_failed.json', 'w') as fout:
        premise_list, hypothesis_list, label_list = [], [], []
        for data in multihop_dataset:
            source = data["text_a"]
            hypothesis = data["text_b"][0]
            label = data["orig_label"]
            premise_list.append(source)
            hypothesis_list.append(hypothesis)
            label_list.append(label)
        
        batch_size = 16
        incorrect = 0
        for i in tqdm(range(0, len(premise_list), batch_size)):
            premises = premise_list[i:i+batch_size]
            hypotheses = hypothesis_list[i:i+batch_size]
            labels = label_list[i:i+batch_size]

            inputs = tokenizer(
                list(zip(premises, hypotheses)),
                max_length=max_length,
                return_token_type_ids=True,
                truncation=True,
                padding=True
            )
            
            input_ids = torch.tensor(inputs['input_ids']).to(device)
            attention_masks = torch.tensor(inputs['attention_mask']).to(device)
            labels = torch.tensor(labels).to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
            logits = outputs.logits

            # Convert logits to predictions
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            labels = [l.item() for l in labels]
            predictions = [0 if p.item() == 1 or p.item() == 2 else 1 for p in predictions] # 0 for entailment, 1 for neutral and 2 for contradiction.
            for prediction, label, premise, hypothesis in zip(predictions, labels, premises, hypotheses):
                if int(prediction) != int(label):
                    incorrect += 1
                    cur_data = {
                        "task": "bin_grounding",
                        "text_a": premise,
                        "text_b": [hypothesis],
                        "text_c": [],
                        "orig_label": label
                    }
                    json.dump(cur_data, fout)
                    fout.write('\n')
    print(f"Total incorrect: {incorrect}")
    print(f"Total data: {len(premise_list)}")

def get_training_Data_kg_noisy_facts(multihop_dataset):
    with open('/home/derenlei/data_synthetic/musique_minhop3_kg_noisy_facts_constructed_all.json', 'w') as fout:
        for data in tqdm(multihop_dataset):
            if "corrupted_fact" not in data:
                continue
                
            # corrupted_fact_sentences list
            corrupted_fact = data["corrupted_fact"]
            hypothesis = data["declarative_sentence"]

            source = []
            # positive sample (grounded)
            # if random.uniform(0, 1) > 0.5:
            for sentence in data["paragraphs"]:
                source.append(sentence["paragraph_text"])

            cur_data = {
                "task": "bin_grounding",
                "text_a": "\n".join(source),
                "text_b": [hypothesis],
                "text_c": [],
                "orig_label": 1
            }
            json.dump(cur_data, fout)
            fout.write('\n')
            # # negative sample (not grounded)
            # else:
            # replace facts in content with corrupted facts
            
            count = 0
            for sentence in data["paragraphs"]:
                if sentence["is_supporting"]:
                    source.append(corrupted_fact[count])
                    count += 1
                else:
                    source.append(sentence["paragraph_text"])
            assert count == len(corrupted_fact), f"{count} != {len(corrupted_fact)}"

            cur_data = {
                "task": "bin_grounding",
                "text_a": "\n".join(source),
                "text_b": [hypothesis],
                "text_c": [],
                "orig_label": 0
            }
            json.dump(cur_data, fout)
            fout.write('\n')

# scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./minicheck_ckpts')
# gpt_caller = GPTCaller(modelname="gpt-4-o")

# file_path = "/home/derenlei/data_public/musique/musique_full_v1.0_train.jsonl"
# file_path = "/home/derenlei/preprocessing/musique_train_minhop3_declarative.json"
file_path = "/home/derenlei/data_synthetic/musique_full_train_minhop3.json"

# rm entity approach
# file_path = "/home/derenlei/data_synthetic/musique_minhop3_relevant_triples.json"
# file_path = "/home/derenlei/data_synthetic/musique_minhop3_kg_noisy_facts.json"
# file_path = "/home/derenlei/data_synthetic/musique_minhop3_kg_noisy_facts_constructed.json" # each data creates either a positive or negative sample
# file_path = "/home/derenlei/data_synthetic/musique_minhop3_kg_noisy_facts_constructed_all.json" # each data creates a positive and negative sample

with open(file_path, "r") as file:
    lines = file.readlines()
    multihop_dataset = [json.loads(line) for line in lines]#[0]
# multihop_dataset = [data for data in multihop_dataset[0] if "2hop" not in data["id"]]
# multihop_dataset = [data for data in multihop_dataset if data.get("answerable")]

# get_declaratve_sentence(multihop_dataset, gpt_caller)
# get_training_data(multihop_dataset[0])
filter_hard_data_by_model(multihop_dataset)

# extract_triple(multihop_dataset, gpt_caller)

# print(len(multihop_dataset))
# get_gold_grounding_data_rm_entity(multihop_dataset, gpt_caller)
# get_training_Data_kg_noisy_facts(multihop_dataset)
# filter_hard_data_by_model(multihop_dataset)

# for data in multihop_dataset:
#     if data["question"] != "Where did the leader of the largest European country after the collapse of the country that denied anything more than an advisory role in the Korean war die?":
#         continue
#     for paragraph in data["paragraphs"]:
#         # if "Soviet Union" not in paragraph["paragraph_text"]:
#         #     continue
#         if paragraph["is_supporting"]:
#             print(paragraph["paragraph_text"])
#             print(paragraph["is_supporting"])
#     print(data["answerable"])
#     print("----------------")

# for data in multihop_dataset[1:]:
#     # print relevant triples
#     if "relevant_triples" in data:
#         # print triples one by one
#         for triple in data["relevant_triples"]:
#             print(triple)
#     print(data["declarative_sentence"])
#     print(data["question"])
#     # print document where it is an suppoerting fact
#     if "paragraphs" in data:
#         i = 0
#         for paragraph in data["paragraphs"]:
#             if paragraph["is_supporting"]:
#                 print(paragraph["paragraph_text"])
#                 print(data["corrupted_fact"][i])
#                 print("----------------")
#                 i += 1
#     break
# print(multihop_dataset[0])
