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
    # activations[:,:,:] = sparse_autoencoder(activations[:,:,:])[0]  # 不包含image的输出， 经过sae, sae特征干预与否，取决于sparse_autoencoder.py 模块， 但是所有的文本部(问题+答案))都经历了sae本身reconstruct的loss
    # activations[:,-1,:] = sparse_autoencoder(activations[:,-1,:])[0]   #可以含有图片也可以不含有图片， 经过sae, sae特征干预与否，取决于sparse_autoencoder.py 模块，只有回答部分经历了sae本身reconstruct的loss
    
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
sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/c19ed8ba9460def36b0931e2555bbe0b0893ebf3/724984992_pre_trained_llava_sae_language_model_65536_update.pt"
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
forget_dataset = load_dataset(dataset_path, "forget_10")['train']



"""
Name: Experiment 1
Goal: Explore the relationship: Do tokens with the same name exhibit different SAE features and values across different language contexts?
Action: We extracted the name associated with each data in the forget_dataset and generated questions using the format: *"What is {name}'s annual salary?"*
We test whether the correct answer number change when we interfere the selected sae features.
"""

max_token = 50
count = 0
i = 0
i_list = []
for forget_index in range(len(forget_dataset)):
    forget_image = forget_dataset[forget_index]['image']
    forget_biography = forget_dataset[forget_index]['biography']
    name = json.loads(forget_biography)["Name"]

    question = f"What is {name}'s annual salary?"

    forget_conversation = conversation_form(question)
    output_texts = generate_image_text(model, forget_conversation, forget_image, max_token)
    
    if 'Annual Salary' in json.loads(forget_biography):
        real_salary = json.loads(forget_biography)['Annual Salary']
        if real_salary.lower() in output_texts.lower():
            i_list.append(i)
            count += 1
        # print(forget_biography)
        print(f"real_salary: {real_salary}")
        print("-----------------")
        print(output_texts)
        print('\n')
        i += 1
print(f"count: {count}")
print(i_list)