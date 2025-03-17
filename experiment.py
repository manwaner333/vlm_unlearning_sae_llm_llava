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


seed = 42 
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/709a7660740cb17bf400ef262233b5a081177ea8/32954368_sae_language_model_32768.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/2c58bb442597ff8270f6f7b29d40400a952bd759/65907712_sae_language_model_32768.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/2c58bb442597ff8270f6f7b29d40400a952bd759/98862080_sae_language_model_32768.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/be2c531fab165b62189ec31bf679b614fb1c65e5/263630848_sae_language_model_32768.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/3c7f08e0a4759c2a86c38e07930f880ef59e498c/65912400_sae_language_model_65536.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/9c619c78842697dff3d581f21fb1bf923adef0a7/32956200_sae_language_model_65536.pt"
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/c14aea2f8281191bfb6c01620a00ff3c723416ca/65908116_sae_language_model_65536.pt"
sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/a50fafec11684ca99d9db96a593489b071e9e285/230677488_sae_language_model_65536.pt"
dataset_path = "MLLMMU/MLLMU-Bench"


# 尝试加载更多的数据
sparse_autoencoder, model = load_sae_model(sae_path)
sparse_autoencoder.eval()
dataset = load_dataset(dataset_path, "Retain_Set")['train']
print(f"Total data quantity: {len(dataset)}")
index = 3
image = dataset[index]['image']

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
 
biography = dataset[index]['biography']
question = dataset[index]['question']
answer = dataset[index]['answer']
Classification_Task = dataset[index]['Classification_Task']
Generation_Task = dataset[index]['Generation_Task']
Mask_Task = dataset[index]['Mask_Task']

# image_file = "great_wall.jpg"
# image = Image.open(image_file)

conversation = conversation_form(question)
prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
model_inputs = model.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
input_ids = model_inputs.input_ids
attention_mask = model_inputs.attention_mask
pixel_values = model_inputs.pixel_values
aspect_ratio_ids = model_inputs.aspect_ratio_ids  
aspect_ratio_mask = model_inputs.aspect_ratio_mask
cross_attention_mask = model_inputs.cross_attention_mask
generated_ids = input_ids.clone()

def sae_hook(activations):
    activations[:,-1,:] = sparse_autoencoder(activations[:,-1,:])[0]
    # activations[:,-1,:] = activations[:,-1,:]
    return (activations,)
        
sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)] 
max_token = 100
print("test case:")
for ele in range(max_token):
    outputs = model.run_with_hooks(
        sae_hooks,
        return_type='output',
        input_ids=generated_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        aspect_ratio_ids=aspect_ratio_ids,
        aspect_ratio_mask=aspect_ratio_mask,
        cross_attention_mask=cross_attention_mask,
        # image_sizes=image_sizes,
    )
    logits = outputs.logits[:, -1, :]  
    next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    new_mask = torch.ones((attention_mask.shape[0], 1), device=sparse_autoencoder.cfg.device, dtype=attention_mask.dtype)
    attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
    if cross_attention_mask is not None:
        size_0, size_1, size_2, size_3 = cross_attention_mask.shape
        new_cross_attention_mask = torch.ones(
            (size_0, 1, size_2, size_3),
            device=cross_attention_mask.device
        )
        cross_attention_mask = torch.cat([cross_attention_mask, new_cross_attention_mask], dim=1)
    torch.cuda.empty_cache()

print("Generated Answer:")
output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(output_texts)

print("Original Answer:")
print(answer)














### 可用的测试方法

# sparse_autoencoder, model = load_sae_model(sae_path)

# sparse_autoencoder.eval()
# image_file = "image1.jpg"
# raw_image = Image.open(image_file)

# conversation = [{"role": "user", "content": [
#     {"type": "image"},
#     {"type": "text", "text": "Please describe this image?"}
# ]}]

# prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
# model_inputs = model.processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
# input_ids = model_inputs.input_ids
# attention_mask = model_inputs.attention_mask
# pixel_values = model_inputs.pixel_values
# aspect_ratio_ids = model_inputs.aspect_ratio_ids  # for Llama-3.2-11B-Vision-Instruct
# aspect_ratio_mask = model_inputs.aspect_ratio_mask
# cross_attention_mask = model_inputs.cross_attention_mask
# generated_ids = input_ids.clone()

# def sae_hook1(activations):
#     activations[:,-1,:] = sparse_autoencoder(activations[:,-1,:])[0]
#     # activations[:,-1,:] = 0.5 * sparse_autoencoder(activations[:,-1,:])[0] + 0.5 * activations[:,-1,:]
#     # activations[:,:,:] = activations[:,:,:]
#     # print(activations.shape)
#     return (activations,)
        
# sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook1, return_module_output=True)] 
# max_token = 100
# print("test case:")
# for ele in range(max_token):
#     # print(ele)
#     outputs = model.run_with_hooks(
#         sae_hooks,
#         return_type='output',
#         input_ids=generated_ids,
#         attention_mask=attention_mask,
#         pixel_values=pixel_values,
#         aspect_ratio_ids=aspect_ratio_ids,
#         aspect_ratio_mask=aspect_ratio_mask,
#         cross_attention_mask=cross_attention_mask,
#         # image_sizes=image_sizes,
#     )
#     logits = outputs.logits[:, -1, :]  
#     next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
#     generated_ids = torch.cat([generated_ids, next_token], dim=-1)
#     new_mask = torch.ones((attention_mask.shape[0], 1), device=sparse_autoencoder.cfg.device, dtype=attention_mask.dtype)
#     attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
#     if cross_attention_mask is not None:
#         size_0, size_1, size_2, size_3 = cross_attention_mask.shape
#         new_cross_attention_mask = torch.ones(
#             (size_0, 1, size_2, size_3),
#             device=cross_attention_mask.device
#         )
#         cross_attention_mask = torch.cat([cross_attention_mask, new_cross_attention_mask], dim=1)

#     torch.cuda.empty_cache()
# output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(output_texts)
# sparse_autoencoder.train()

