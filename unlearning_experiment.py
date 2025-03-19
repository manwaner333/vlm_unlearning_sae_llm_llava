import os
import sys
import torch
import wandb
import json
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from functools import partial
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import get_feature_data, FeatureData
import io

import os
import sys
import torch
import wandb
import json
import pickle
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from tqdm import tqdm
from functools import partial
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from torch.utils.data import DataLoader 
from datasets import Dataset, Features, Value
from datasets import Image as dataset_Image 
import json
from tqdm import tqdm, trange
import torch
import random
import numpy as np

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from vit_sae_analysis.dashboard_fns import get_feature_data
from sae_training.utils import ViTSparseAutoencoderSessionloader
from sae_training.hooked_vit import HookedVisionTransformer, Hook

import os
import sys
import torch
import wandb
import json
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from tqdm import tqdm
from functools import partial
from vit_sae_analysis.dashboard_fns import get_feature_data   # FeatureData

import gzip
import json
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
from datasets import load_dataset
from tqdm import trange
from eindex import eindex
from IPython.display import HTML, display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor, topk
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_activations_store import ViTActivationsStore
import torchvision.transforms as transforms
from PIL import Image
from sae_training.utils import ViTSparseAutoencoderSessionloader
import shutil


from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import get_feature_data    # FeatureData
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from vit_sae_analysis.dashboard_fns import get_feature_data     # FeatureData
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import ViTSparseAutoencoderSessionloader
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from plotly import express as xp
import torch
import plotly.io as pio
from typing import Union, List, Optional
import torch


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
    
    activations = activations[:,577:,:]

    return activations



def extratc_forget_retain_sae_feature_info(forget_dataset, retain_dataset, forget_index, retain_index, k, hook_name):

    forget_image = forget_dataset[forget_index]['image']
    forget_biography = forget_dataset[forget_index]['biography']
    forget_conversation = conversation_form(forget_biography)
    forget_prompt = model.processor.apply_chat_template(forget_conversation, add_generation_prompt=True)
    forget_inputs = model.processor(images=forget_image, text=forget_prompt, return_tensors='pt').to(0, torch.float16)
    forget_model_activations = get_model_activations(model, forget_inputs, sparse_autoencoder.cfg)

    forget_sae_activations = sparse_autoencoder.run_with_cache(forget_model_activations)[1][hook_name]
    forget_summed_sae_activations = forget_sae_activations[0].mean(dim=0, keepdim=True) 
    forget_values_ele, forget_indices_ele = torch.topk(forget_summed_sae_activations, k, dim=1)


    retain_image = retain_dataset[retain_index]['image']
    retain_biography = retain_dataset[retain_index]['biography']
    retain_conversation = conversation_form(retain_biography)
    retain_prompt = model.processor.apply_chat_template(retain_conversation, add_generation_prompt=True)
    retain_inputs = model.processor(images=retain_image, text=retain_prompt, return_tensors='pt').to(0, torch.float16)
    retain_model_activations = get_model_activations(model, retain_inputs, sparse_autoencoder.cfg)

    retain_sae_activations = sparse_autoencoder.run_with_cache(retain_model_activations)[1][hook_name]
    retain_summed_sae_activations = retain_sae_activations[0].mean(dim=0, keepdim=True) 
    retain_values_ele, retain_indices_ele = torch.topk(retain_summed_sae_activations, k, dim=1)

    return forget_values_ele, forget_indices_ele, retain_values_ele, retain_indices_ele


def formulate_prompt_with_options(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt


### load sparse autoencoder
sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/c56dd1601694cfb7a43202199b0f25a4b617a83b/32954364_pre_trained_llava_sae_language_model_65536.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/804e09f0776b759f6c0949466f9c8ff218e314ea/65908116_pre_trained_llava_sae_language_model_65536.pt"
sparse_autoencoder, model = load_sae_model(sae_path)
sparse_autoencoder.eval()

### load dataset
dataset_path = "MLLMMU/MLLMU-Bench"
forget_dataset = load_dataset(dataset_path, "forget_10")['train']
retain_dataset = load_dataset(dataset_path, "retain_90")['train']

### generate sae features
# hook_name = "hook_hidden_post"
# k = 30
# forget_values = []
# forget_indices = []
# retain_values = []
# retain_indices = []

# for forget_index in range(3, 4):
#     retain_index = random.randint(0, 450)
#     forget_values_ele, forget_indices_ele, retain_values_ele, retain_indices_ele = extratc_forget_retain_sae_feature_info(forget_dataset, retain_dataset, forget_index, retain_index, k, hook_name)
#     forget_values.extend(forget_values_ele[0].cpu().numpy().tolist())
#     forget_indices.extend(forget_indices_ele[0].cpu().numpy().tolist())
#     retain_values.extend(retain_values_ele[0].cpu().numpy().tolist())
#     retain_indices.extend(retain_indices_ele[0].cpu().numpy().tolist())

# print(len(forget_values))
# print(len(forget_indices))
# print(len(retain_values))
# print(len(retain_indices))


# with open("dataset/forget_values.pkl", "wb") as f:
#     pickle.dump(forget_values, f)

# with open("dataset/forget_indices.pkl", "wb") as f:
#     pickle.dump(forget_indices, f)

# with open("dataset/retain_values.pkl", "wb") as f:
#     pickle.dump(retain_values, f)

# with open("dataset/retain_indices.pkl", "wb") as f:
#     pickle.dump(retain_indices, f)
    

# ## choose sae_features
# with open("dataset/forget_values.pkl", "rb") as f:
#     forget_values = pickle.load(f)

# with open("dataset/forget_indices.pkl", "rb") as f:
#     forget_indices = pickle.load(f)

# with open("dataset/retain_values.pkl", "rb") as f:
#     retain_values = pickle.load(f)

# with open("dataset/retain_indices.pkl", "rb") as f:
#     retain_indices = pickle.load(f)
    

# # forget
# forget_index_value_dict = {}
# for idx, val in zip(forget_indices, forget_values):
#     if idx in forget_index_value_dict:
#         forget_index_value_dict[idx] = max(forget_index_value_dict[idx], val)
#     else:
#         forget_index_value_dict[idx] = val
# unique_forget_indices = list(forget_index_value_dict.keys())
# unique_forget_values = list(forget_index_value_dict.values())
# print(len(unique_forget_indices))

# # retain
# retain_index_value_dict = {}
# for idx, val in zip(retain_indices, retain_values):
#     if idx in retain_index_value_dict:
#         retain_index_value_dict[idx] = max(retain_index_value_dict[idx], val)
#     else:
#         retain_index_value_dict[idx] = val
# unique_retain_indices = list(retain_index_value_dict.keys())
# unique_retain_values = list(retain_index_value_dict.values())
# print(len(unique_retain_indices))

# retain_set = set(unique_retain_indices)
# exclusive_forget_indices = []
# exclusive_forget_values = []
# for idx, val in zip(unique_forget_indices, unique_forget_values):
#     if idx not in retain_set:
#         exclusive_forget_indices.append(idx)
#         exclusive_forget_values.append(val)

# print(exclusive_forget_indices)
# print(exclusive_forget_values)


### unlearning
def sae_hook(activations):
    activations[:,-1,:] = sparse_autoencoder(activations[:,-1,:])[0]
    # activations[:,-1,:] = activations[:,-1,:]
    return (activations,)

def generate_image_text(model, conversation, image, max_token):
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
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            new_mask = torch.ones((attention_mask.shape[0], 1), device=sparse_autoencoder.cfg.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            torch.cuda.empty_cache()

        output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output_texts

def generate_text(model, conversation, max_token):
    with torch.no_grad():
        prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
        model_inputs = model.processor(text=prompt, return_tensors='pt').to(0, torch.float16)
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        # pixel_values = model_inputs.pixel_values
        generated_ids = input_ids.clone()
                
        sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)] 
        print("test case:")
        for ele in range(max_token):
            outputs = model.run_with_hooks(
                sae_hooks,
                return_type='output',
                input_ids=generated_ids,
                attention_mask=attention_mask,
                # pixel_values=pixel_values,
                # image_sizes=image_sizes,
            )
            logits = outputs.logits[:, -1, :]  
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            new_mask = torch.ones((attention_mask.shape[0], 1), device=sparse_autoencoder.cfg.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            torch.cuda.empty_cache()

        output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output_texts
        
def evaluate_classification(classification_task, image, model):
    print("################################## Classification Task Starts ##############################################")
    print("################################## Image Textual Questions ##############################################")
    for ele in classification_task['Image_Textual_Questions']:
        correct_answer = ele['Correct_Answer']
        options = ele['Options']
        question = ele['Question']
        
        # combined_question = f"You will be given a Question and multiple Options. Please choose the correct answer from the Options to answer Question. The answer form is A, B, C or D. \n Question: {question} \n Options: {options}"
        # combined_question = f"""You will be given a Question and multiple Options. Please choose the correct answer from the given Options.
        
        # The answer must be one of the following: A, B, C, or D. Do not provide explanations, just output a single letter.
        
        # Respond only with one letter: A, B, C, or D.

        # Question: {question}
        # Options: {options}
        # Answer:
        # """
        
        question_with_options = formulate_prompt_with_options(question, options)
        combined_question = (f"{question_with_options}"
                      f"Just give ONE letter representing the answer directly.\nAnswer:")
        conversation = conversation_form(combined_question)
        max_token = 2
        output_texts = generate_image_text(model, conversation, image, max_token)
        
        print("Generated Answer:")
        print(output_texts)

        print("Original Answer:")
        print(correct_answer)
        
    print("################################## Textual Questions ##############################################")
    for ele in Classification_Task['Pure_Text_Questions']:
        correct_answer = ele['Correct_Answer']
        options = ele['Options']
        question = ele['Question']
        
        question_with_options = formulate_prompt_with_options(question, options)
        combined_question = (f"{question_with_options}"
                      f"Just give ONE letter representing the answer directly.\nAnswer:")
        conversation = conversation_only_text_form(combined_question)
        max_token = 2
        output_texts =  generate_text(model, conversation, max_token)
        
        print("Generated Answer:")
        print(output_texts)

        print("Original Answer:")
        print(correct_answer)

def evaluate_generation(generation_task, image, model):
    print("################################## Generation Task Starts ##############################################")
    for ele in generation_task:
        ground_truth = ele['Ground_Truth']
        type = ele['Type']
        question = ele['Question']
        
        combined_question = f"""Answer the following question based on your trained knowledge in one sentence accurately in ENGLISH.
        Question: {question}
        Answer:
        """
        max_token = 20
        if type == 'Image_Textual':
            conversation = conversation_form(combined_question)
            output_texts = generate_image_text(model, conversation, image, max_token)
        else:
            conversation = conversation_only_text_form(combined_question)
            output_texts =  generate_text(model, conversation, max_token)
        
        print("Generated Answer:")
        print(output_texts)

        print("Original Answer:")
        print(ground_truth)
        
def evaluate_fill_blank(mask_task, image, model):
    print("################################## Fill-in-the-blank Task Starts ##############################################")
    for ele in mask_task:
        question = ele["Question"]
        ground_truth = ele["Ground_Truth"]
        question_type = ele["Type"]
        
        combined_question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
        max_token = 5
        if question_type == "Image_Textual":
            conversation = conversation_form(combined_question)
            output_texts = generate_image_text(model, conversation, image, max_token)
        else:
            conversation = conversation_only_text_form(combined_question)
            output_texts =  generate_text(model, conversation, max_token)
        
        print("Generated Answer:")
        print(output_texts)

        print("Original Answer:")
        print(ground_truth)
    
    
for index in [3]:  # range(0, 50):
    print(f"index: {index}")
    image = forget_dataset[index]['image']
    biography = forget_dataset[index]['biography']
    question = forget_dataset[index]['question']
    answer = forget_dataset[index]['answer']
    Classification_Task = forget_dataset[index]['Classification_Task']
    Generation_Task = forget_dataset[index]['Generation_Task']
    Mask_Task = forget_dataset[index]['Mask_Task']
    
    # evaluate_classification(Classification_Task, image, model)
    # evaluate_generation(Generation_Task, image, model)
    evaluate_fill_blank(Mask_Task, image, model)
        
       