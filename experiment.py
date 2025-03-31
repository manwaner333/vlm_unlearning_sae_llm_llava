import os
import sys
import io
import gzip
import json
import time
import pickle
import shutil
import random
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, topk
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, Features, Value
from datasets import Image as dataset_Image
from PIL import Image
import requests
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm, trange
from rich import print as rprint
from rich.table import Table
from IPython.display import HTML, display
import einops
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import spacy
import torchvision.transforms as transforms
from torchvision import transforms, datasets
from torchvision.utils import save_image
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration
)
from sae_training.config import ViTSAERunnerConfig
from sae_training.utils import (
    LMSparseAutoencoderSessionloader,
    ViTSparseAutoencoderSessionloader
)
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.vit_activations_store import ViTActivationsStore
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer

from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import get_feature_data  # FeatureData
from vit_sae_analysis.dashboard_fns import get_feature_data    # FeatureData
# nlp = spacy.load("en_core_web_sm")


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.path.append("..")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

# "Please describe the content of this image." + 
# "Please tell me some information about the building in the picture."

def conversation_form(key):
    conversation = [
        {"role": "user",
        "content": [
            {"type": "text", "text": key},
            {"type": "image"},
            ],
        },
    ]
    return conversation

def conversation_only_text_form(key):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": key}
            ],
        },
    ]
    return conversation
    
def load_sae_model(sae_path):
    sae_path = sae_path
    loaded_object = torch.load(sae_path)
    cfg = loaded_object['cfg']
    state_dict = loaded_object['state_dict']

    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.load_state_dict(state_dict)
    sparse_autoencoder.eval()

    loader = ViTSparseAutoencoderSessionloader(cfg)
    model = loader.get_model(cfg.model_name)
    model.to(cfg.device)
    
    return sparse_autoencoder, model


def get_model_activations(model, inputs, cfg):
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]

    activations = model.run_with_cache(
        list_of_hook_locations,
        **inputs,
    )[1][(block_layer, module_name)]
    
    # activations = activations[:,:,:]
    activations = activations[:,575:,:]

    return activations

def conversation_only_text_form(key):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": key}
            ],
        },
    ]
    return conversation


def sae_hook(activations):
    activations[:,575:,:] = sparse_autoencoder(activations[:,575:,:])[0] # 包含image的输出， 经过sae, sae特征干预与否，取决于sparse_autoencoder.py 模块， 但是所有的文本部(问题+答案))都经历了sae本身reconstruct的loss
    activations[:,:,:] = sparse_autoencoder(activations[:,:,:])[0]  # 不包含image的输出， 经过sae, sae特征干预与否，取决于sparse_autoencoder.py 模块， 但是所有的文本部(问题+答案))都经历了sae本身reconstruct的loss
    activations[:,-1,:] = sparse_autoencoder(activations[:,-1,:])[0]   #可以含有图片也可以不含有图片， 经过sae, sae特征干预与否，取决于sparse_autoencoder.py 模块，只有回答部分经历了sae本身reconstruct的loss
    
    # activations[:,:,:] = activations[:,:,:]  # 未经过sae的任何处理
    # activations[:,-1,:] = activations[:,-1,:] # 未经过sae的任何处理
    
    return (activations,)

def generate_image_text(model, conversation, image, max_token):
    sentence_end_pattern = re.compile(r"[.?!]\s*$")
    with torch.no_grad():
        prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
        model_inputs = model.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        pixel_values = model_inputs.pixel_values
        generated_ids = input_ids.clone()
                
        sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)] 
        print("test case:")
        for ele in range(max_token):
            outputs = model.run_with_hooks(
                sae_hooks,
                return_type='output',
                input_ids=generated_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                # image_sizes=image_sizes,
            )
            logits = outputs.logits[:, -1, :]  
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            
            if next_token == model.model.config.eos_token_id:
                break
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            new_mask = torch.ones((attention_mask.shape[0], 1), device=sparse_autoencoder.cfg.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            
            decoded_text = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # print(decoded_text)
            if sentence_end_pattern.search(decoded_text):
                break
    
            torch.cuda.empty_cache()

        output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return output_texts


### load sparse_autoencoder and model
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/424fb7f12fba943f7b029262f6fb1d9c2f0f3262/131815620_pre_trained_llava_sae_language_model_65536_update.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/9307c4400294c174480ba20955c992408f6f4413/395446248_pre_trained_llava_sae_language_model_65536_update.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/c19ed8ba9460def36b0931e2555bbe0b0893ebf3/724984992_pre_trained_llava_sae_language_model_65536_update.pt"
sae_path = "/home/coder/geng/vlm_unlearning_sae_llm_llava/checkpoints/wrif7maq_1/724984992_sparse_autoencoder_LLaVA_Vanilla_16_resid_65536.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/003628e7ac7aff3a437f92d691bfcf8be7799a9e/757938744_pre_trained_llava_sae_language_model_65536_update.pt"
loaded_object = torch.load(sae_path)
cfg = loaded_object['cfg']
state_dict = loaded_object['state_dict']
sparse_autoencoder = SparseAutoencoder(cfg)
sparse_autoencoder.load_state_dict(state_dict)
sparse_autoencoder.eval()
loader = ViTSparseAutoencoderSessionloader(cfg)
model = loader.get_model(cfg.model_name)
model.to(cfg.device)

# another mothod for loading  sparse_autoencoder and model
sparse_autoencoder, model = load_sae_model(sae_path)
sparse_autoencoder.eval()

### load dataset
dataset_path = "MLLMMU/MLLMU-Bench"
# forget_dataset = load_dataset(dataset_path, "forget_10")['train']
# retain_dataset = load_dataset(dataset_path, "retain_90")['train']
forget_dataset = load_dataset(dataset_path, "retain_90")['train']

"""
Name: Experiment 1
Goal: Explore the relationship: Do tokens with the same name exhibit different SAE features and values across different language contexts?
Action: We extract plain-text questions from the Mask_Task and Classification_Task datasets that contain name information. 
For each token in the names, we retrieve the top-k associated SAE features and their corresponding values. In this experiment, the input contains "image"
"""

# hook_name = "hook_hidden_post"
# k = 5
# with open("dataset/token_infor_retain.json", "w", encoding="utf-8") as f:  
#     for forget_index in range(len(forget_dataset)):
#         forget_image = forget_dataset[forget_index]['image']
#         forget_biography = forget_dataset[forget_index]['biography']
#         name = json.loads(forget_biography)["Name"]
#         name_input_ids = model.processor.tokenizer(name, return_tensors="pt")["input_ids"][0].detach().cpu().numpy()
#         name_tokens = [model.processor.tokenizer.decode(token_id, skip_special_tokens=False) for token_id in name_input_ids]
#         print(name_tokens)
        
#         Classification_Task = forget_dataset[forget_index]['Classification_Task']
#         Generation_Task = forget_dataset[forget_index]['Generation_Task']
#         Mask_Task = forget_dataset[forget_index]['Mask_Task']
        
#         for ele in Mask_Task:
#             question = ele["Question"]
#             ground_truth = ele["Ground_Truth"]
#             question_type = ele["Type"]

#             if question_type == "Pure_Text":
#                 forget_conversation = conversation_form(question)
#                 forget_prompt = model.processor.apply_chat_template(forget_conversation, add_generation_prompt=True)
#                 forget_inputs = model.processor(images=forget_image, text=forget_prompt, return_tensors='pt').to(0, torch.float16)
#                 forget_model_activations = get_model_activations(model, forget_inputs, sparse_autoencoder.cfg)

#                 input_ids = model.processor.tokenizer(forget_prompt, return_tensors="pt")["input_ids"][0].detach().cpu().numpy()

#                 tokens = [model.processor.tokenizer.decode(token_id) for token_id in input_ids]

#                 forget_sae_activations = sparse_autoencoder.run_with_cache(forget_model_activations)[1][hook_name][0]
#                 values, indices = torch.topk(forget_sae_activations, k, dim=1)

#                 res = {}
#                 for i in range(len(tokens)):
#                     token = tokens[i]
#                     if token in name_tokens and token != '<s>':
#                         print(name)
#                         print(f"token: {token}")
#                         indice = indices[i]
#                         value = values[i]
#                         res[token] = {"name": name, "question:": question, "indice": indice.tolist(), "value": value.tolist()}
                
#                 json.dump(res, f, ensure_ascii=False)
#                 f.write("\n")
        
#         for ele in Classification_Task['Pure_Text_Questions']:
#             correct_answer = ele['Correct_Answer']
#             options = ele['Options']
#             question = ele['Question']
            
#             forget_conversation = conversation_form(question)
#             forget_prompt = model.processor.apply_chat_template(forget_conversation, add_generation_prompt=True)
#             forget_inputs = model.processor(images=forget_image, text=forget_prompt, return_tensors='pt').to(0, torch.float16)
#             forget_model_activations = get_model_activations(model, forget_inputs, sparse_autoencoder.cfg)

#             input_ids = model.processor.tokenizer(forget_prompt, return_tensors="pt")["input_ids"][0].detach().cpu().numpy()

#             tokens = [model.processor.tokenizer.decode(token_id, skip_special_tokens=False) for token_id in input_ids]

#             forget_sae_activations = sparse_autoencoder.run_with_cache(forget_model_activations)[1][hook_name][0]
#             values, indices = torch.topk(forget_sae_activations, k, dim=1)

#             res = {}
#             for i in range(len(tokens)):
#                 token = tokens[i]
#                 if token in name_tokens and token != '<s>':  # [-1]
#                     print(name)
#                     print(f"token: {token}")
#                     indice = indices[i]
#                     value = values[i]
#                     res[token] = {"name": name, "question:": question, "indice": indice.tolist(), "value": value.tolist()}
            
#             json.dump(res, f, ensure_ascii=False)
#             f.write("\n")
           

"""
Name: Experiment 2
Goal: Explore the relationship: Do tokens with the same name exhibit different SAE features and values across different language contexts?
Action: We extract plain-text questions from the Mask_Task and Classification_Task datasets that contain name information. 
For each token in the names, we retrieve the top-k associated SAE features and their corresponding values. The diference with Experiment 1 is that the input donesn't contain "image"
"""


# with open("dataset/token_infor_retain.json", "w", encoding="utf-8") as f: 
#     for forget_index in range(len(forget_dataset)):
#         forget_image = forget_dataset[forget_index]['image']
#         forget_biography = forget_dataset[forget_index]['biography']
#         name = json.loads(forget_biography)["Name"]
#         # name = name.replace("<s>", "").strip()
#         name_input_ids = model.processor.tokenizer(name, return_tensors="pt")["input_ids"][0].detach().cpu().numpy()
#         name_tokens = [model.processor.tokenizer.decode(token_id, skip_special_tokens=False) for token_id in name_input_ids]
#         print(name_tokens)
#         # print(forget_biography)
        
#         Classification_Task = forget_dataset[forget_index]['Classification_Task']
#         Generation_Task = forget_dataset[forget_index]['Generation_Task']
#         Mask_Task = forget_dataset[forget_index]['Mask_Task']
        
#         for ele in Mask_Task:
#             question = ele["Question"]
#             ground_truth = ele["Ground_Truth"]
#             question_type = ele["Type"]

#             if question_type == "Pure_Text":
#                 forget_conversation = conversation_only_text_form(question)
#                 forget_prompt = model.processor.apply_chat_template(forget_conversation, add_generation_prompt=True)
#                 forget_inputs = model.processor(text=forget_prompt, return_tensors='pt').to(0, torch.float16)
#                 forget_model_activations = get_model_activations(model, forget_inputs, sparse_autoencoder.cfg)

#                 input_ids = model.processor.tokenizer(forget_prompt, return_tensors="pt")["input_ids"][0].detach().cpu().numpy()

#                 tokens = [model.processor.tokenizer.decode(token_id) for token_id in input_ids]  # '<s>'

#                 forget_sae_activations = sparse_autoencoder.run_with_cache(forget_model_activations)[1][hook_name][0]
#                 values, indices = torch.topk(forget_sae_activations, k, dim=1)

#                 res = {}
#                 for i in range(len(tokens)):
#                     token = tokens[i]
#                     if token in name_tokens and token != '<s>':
#                         print(name)
#                         print(f"token: {token}")
#                         indice = indices[i]
#                         value = values[i]
#                         res[token] = {"name": name, "question:": question, "indice": indice.tolist(), "value": value.tolist()}
                
#                 json.dump(res, f, ensure_ascii=False)
#                 f.write("\n")
        
#         for ele in Classification_Task['Pure_Text_Questions']:
#             correct_answer = ele['Correct_Answer']
#             options = ele['Options']
#             question = ele['Question']
            
#             forget_conversation = conversation_only_text_form(question)
#             forget_prompt = model.processor.apply_chat_template(forget_conversation, add_generation_prompt=True)
#             forget_inputs = model.processor(text=forget_prompt, return_tensors='pt').to(0, torch.float16)
#             forget_model_activations = get_model_activations(model, forget_inputs, sparse_autoencoder.cfg)

#             input_ids = model.processor.tokenizer(forget_prompt, return_tensors="pt")["input_ids"][0].detach().cpu().numpy()

#             tokens = [model.processor.tokenizer.decode(token_id, skip_special_tokens=False) for token_id in input_ids]

#             forget_sae_activations = sparse_autoencoder.run_with_cache(forget_model_activations)[1][hook_name][0]
#             values, indices = torch.topk(forget_sae_activations, k, dim=1)

#             res = {}
#             for i in range(len(tokens)):
#                 token = tokens[i]
#                 if token in name_tokens and token != '<s>':  # [-1]
#                     print(name)
#                     print(f"token: {token}")
#                     indice = indices[i]
#                     value = values[i]
#                     res[token] = {"name": name, "question:": question, "indice": indice.tolist(), "value": value.tolist()}
            
#             json.dump(res, f, ensure_ascii=False)
#             f.write("\n")

"""
Name: Experiment 3
Goal: Explore the relationship: Do tokens with the same name exhibit different SAE features and values across different language contexts?
Action: Based on the above experimental results, We analyze the SAE features that are present in the forget dataset but not in the retain dataset.
"""

# retain_features = {}
# with open("dataset/token_infor_retain.json", "r", encoding="utf-8") as f:
#     for line in f:
#         if line.strip():  # 跳过空行
#             item = json.loads(line)
#             for key in item:
#                 if "indice" in item[key]:
#                     indices = item[key]["indice"][0:3]
#                     for ele in indices:
#                         if ele not in retain_features:
#                             retain_features[ele] = 1
#                         else:
#                             retain_features[ele] += 1
# sorted_retain_features = dict(sorted(retain_features.items(), key=lambda x: x[1], reverse=True))                   
# print(sorted_retain_features)
# qingli = 3


# forget_features = {}
# with open("dataset/token_infor.json", "r", encoding="utf-8") as f:
#     for line in f:
#         if line.strip():  # 跳过空行
#             item = json.loads(line)
#             for key in item:
#                 if "indice" in item[key]:
#                     indices = item[key]["indice"][0:3]
#                     for ele in indices:
#                         if ele not in forget_features:
#                             forget_features[ele] = 1
#                         else:
#                             forget_features[ele] += 1
# sorted_forget_features = dict(sorted(forget_features.items(), key=lambda x: x[1], reverse=True))                   
# print(sorted_forget_features)
# qingli = 3

# token = []
# features = {}
# for forget_ele in sorted_forget_features:
#     if forget_ele not in sorted_retain_features:
#         token.append(forget_ele)
#         features[forget_ele] = sorted_forget_features[forget_ele]

# print(features)
# print(token)



"""
Name: Experiment 4
Goal: Explore whether true forgetting has occurred. Extract the results before and after applying the SAE to explore whether originally correct cases become incorrect, and whether any originally incorrect cases become correct.
Action: We extract plain-text questions from the Mask_Task and Classification_Task datasets that contain name information. 
For each token in the names, we retrieve the top-k associated SAE features and their corresponding values. The diference with Experiment 1 is that the input donesn't contain "image"
"""


# vanilla_info = []
# sae_info = []
# with open("result/llava_1.5_7b_vanilla_model_forget_10/llava_1.5_7b_vanilla_model_forget_10_fill_blank_results_official.json", "r", encoding="utf-8") as f:
#     for line in f:
#         if line.strip(): 
#             item = json.loads(line)
#             vanilla_info.append(item)

# with open("result/llava_1.5_7b_sae_forget_10_3/llava_1.5_7b_sae_forget_10_3_fill_blank_results_official.json", "r", encoding="utf-8") as f:
#     for line in f:
#         if line.strip():  # 跳过空行
#             item = json.loads(line)
#             sae_info.append(item)

# count = 0
# for index in range(len(vanilla_info)):
#     sae_ele = sae_info[index]
#     sae_question_type = sae_ele["question_type"]
#     sae_question = sae_ele["question"]
#     sae_model_answer = sae_ele["model_answer"]
#     sae_ground_truth = sae_ele["ground_truth"]
    
#     if sae_question_type == "Image_Textual":
#         if sae_ground_truth.lower() in sae_model_answer.lower():
#             continue
#         else:
#             vanilla_ele = vanilla_info[index]
#             question_type = vanilla_ele["question_type"]
#             question = vanilla_ele["question"]
#             model_answer = vanilla_ele["model_answer"]
#             ground_truth = vanilla_ele["ground_truth"]
#             if ground_truth.lower() in model_answer.lower():
#                 count += 1
#                 print("SAE:")
#                 print(f"Question: {sae_question}")
#                 print(f"Model Answer: {sae_model_answer}")
#                 print(f"Ground Truth: {sae_ground_truth}")
#                 print(f"Answer: {sae_ground_truth.lower() in sae_model_answer.lower()}")
#                 print("Vanilla:") 
#                 print(f"Question: {question}")
#                 print(f"Model Answer: {model_answer}")
#                 print(f"Ground Truth: {ground_truth}")
#                 print(f"Answer: {ground_truth.lower() in model_answer.lower()}")
#                 print("\n")
# print(f"count: {count}")


"""
Name: Experiment 5
Goal: Explore whether using different subjects with the same object affects the SAE features and values of each token in the object. For example: "zhangsan's salary & lisi's salary" 
Action: We extracted the name associated with each data in the forget_dataset and generated questions using the format: *"What is {name}'s annual salary?"*. 
We extracted the top-k SAE features and values for each token of the word *"salary"*, and analyzed the frequency and corresponding values of all tokens. 
Tokens that appear frequently and have high associated values are assumed to be more closely related to *"salary"*.
"""

# token_number = {}
# token_value = {}
# with open("dataset/salary_token_infor_forget.json", "w", encoding="utf-8") as f:
#     for forget_index in range(len(forget_dataset)):
#         forget_image = forget_dataset[forget_index]['image']
#         forget_biography = forget_dataset[forget_index]['biography']
#         name = json.loads(forget_biography)["Name"]

#         salary = "salary"
#         salary_input_ids = model.processor.tokenizer(salary, return_tensors="pt")["input_ids"][0].detach().cpu().numpy()
#         salary_tokens = [model.processor.tokenizer.decode(token_id, skip_special_tokens=False) for token_id in salary_input_ids]

#         question = f"What is {name}'s annual salary?"

#         forget_conversation = conversation_form(question)
#         forget_prompt = model.processor.apply_chat_template(forget_conversation, add_generation_prompt=True)
#         forget_inputs = model.processor(images=forget_image, text=forget_prompt, return_tensors='pt').to(0, torch.float16)
#         forget_model_activations = get_model_activations(model, forget_inputs, sparse_autoencoder.cfg)

#         input_ids = model.processor.tokenizer(forget_prompt, return_tensors="pt")["input_ids"][0].detach().cpu().numpy()
#         tokens = [model.processor.tokenizer.decode(token_id) for token_id in input_ids]  

#         forget_sae_activations = sparse_autoencoder.run_with_cache(forget_model_activations)[1][hook_name][0]
#         values, indices = torch.topk(forget_sae_activations, k, dim=1)

#         res = {}
#         for i in range(len(tokens)):
#             token = tokens[i]
#             if token == salary_tokens[-1]:
#                 print(f"token: {token}")
#                 indice = indices[i].tolist()
#                 value = values[i].tolist()
#                 res[token] = {"name": name, "question:": question, "indice": indice, "value": value}
#                 for ele_i in range(len(indice)):
#                     ele_indice =  indice[ele_i]
#                     ele_value = value[ele_i]
#                     if ele_indice not in token_number:
#                         token_number[ele_indice] = 1
#                         token_value[ele_indice] = [ele_value]
#                     else:
#                         token_number[ele_indice] += 1
#                         token_value[ele_indice].append(ele_value)
            
#         json.dump(res, f, ensure_ascii=False)
#         f.write("\n")

# res_features = dict(sorted(token_number.items(), key=lambda x: x[1], reverse=True))                   
# print(res_features)
# print(token_value)
