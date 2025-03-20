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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import spacy
import re
nlp = spacy.load("en_core_web_sm")


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



### load sparse autoencoder
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/c56dd1601694cfb7a43202199b0f25a4b617a83b/32954364_pre_trained_llava_sae_language_model_65536.pt"
sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/424fb7f12fba943f7b029262f6fb1d9c2f0f3262/131815620_pre_trained_llava_sae_language_model_65536_update.pt"
sparse_autoencoder, model = load_sae_model(sae_path)
sparse_autoencoder.eval()

### load dataset
dataset_path = "MLLMMU/MLLMU-Bench"
forget_dataset = load_dataset(dataset_path, "forget_10")['train']
retain_dataset = load_dataset(dataset_path, "retain_90")['train']

### generate sae features
hook_name = "hook_hidden_post"
k = 20
forget_values = []
forget_indices = []
retain_values = []
retain_indices = []

for forget_index in range(0, 50):
    retain_index = random.randint(0, 450)
    forget_values_ele, forget_indices_ele, retain_values_ele, retain_indices_ele = extratc_forget_retain_sae_feature_info(forget_dataset, retain_dataset, forget_index, retain_index, k, hook_name)
    forget_values.extend(forget_values_ele[0].cpu().numpy().tolist())
    forget_indices.extend(forget_indices_ele[0].cpu().numpy().tolist())
    retain_values.extend(retain_values_ele[0].cpu().numpy().tolist())
    retain_indices.extend(retain_indices_ele[0].cpu().numpy().tolist())

print(len(forget_values))
print(len(forget_indices))
print(len(retain_values))
print(len(retain_indices))


with open("dataset/forget_values.pkl", "wb") as f:
    pickle.dump(forget_values, f)

with open("dataset/forget_indices.pkl", "wb") as f:
    pickle.dump(forget_indices, f)

with open("dataset/retain_values.pkl", "wb") as f:
    pickle.dump(retain_values, f)

with open("dataset/retain_indices.pkl", "wb") as f:
    pickle.dump(retain_indices, f)
    

## upload sae_features
with open("dataset/forget_values.pkl", "rb") as f:
    forget_values = pickle.load(f)

with open("dataset/forget_indices.pkl", "rb") as f:
    forget_indices = pickle.load(f)

with open("dataset/retain_values.pkl", "rb") as f:
    retain_values = pickle.load(f)

with open("dataset/retain_indices.pkl", "rb") as f:
    retain_indices = pickle.load(f)
    

## forget
forget_index_value_dict = {}
for idx, val in zip(forget_indices, forget_values):
    if idx in forget_index_value_dict:
        forget_index_value_dict[idx] = max(forget_index_value_dict[idx], val)
    else:
        forget_index_value_dict[idx] = val
unique_forget_indices = list(forget_index_value_dict.keys())
unique_forget_values = list(forget_index_value_dict.values())
print(f"number of unique forget sae features: {len(unique_forget_indices)}")

# retain
retain_index_value_dict = {}
for idx, val in zip(retain_indices, retain_values):
    if idx in retain_index_value_dict:
        retain_index_value_dict[idx] = max(retain_index_value_dict[idx], val)
    else:
        retain_index_value_dict[idx] = val
unique_retain_indices = list(retain_index_value_dict.keys())
unique_retain_values = list(retain_index_value_dict.values())
print(f"number of unique retain sae features: {len(unique_retain_indices)}")


retain_set = set(unique_retain_indices)
exclusive_forget_indices = []
exclusive_forget_values = []
for idx, val in zip(unique_forget_indices, unique_forget_values):
    if idx not in retain_set:
        exclusive_forget_indices.append(idx)
        exclusive_forget_values.append(val)

print(exclusive_forget_indices)
print(exclusive_forget_values)


