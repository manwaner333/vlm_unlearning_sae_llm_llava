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

sys.path.append("..")

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from vit_sae_analysis.dashboard_fns import get_feature_data
from sae_training.utils import ViTSparseAutoencoderSessionloader
from sae_training.hooked_vit import HookedVisionTransformer, Hook

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

cfg = ViTSAERunnerConfig(
    
    # Data Generating Function (Model + Training Distibuion)
    class_token = True,
    image_width = 224,
    image_height = 224,
    model_name = "llava-hf/llava-v1.6-vicuna-7b-hf",  # "openai/clip-vit-large-patch14",
    module_name = "resid",
    block_layer = -2,
    dataset_path = "./dataset/select.json",   # "evanarlian/imagenet_1k_resized_256",
    use_cached_activations = False,
    cached_activations_path = None,
    d_in = 4096,  # 1024,
    
    # SAE Parameters
    expansion_factor = 3,  # 64,
    b_dec_init_method = "mean",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    lr_scheduler_name="constantwithwarmup",
    batch_size = 2, # 1024,
    lr_warm_up_steps=500,
    total_training_tokens = 10,  # 20000, # 2_621_440,
    n_batches_in_store = 3, # 15,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_method = None,
    feature_sampling_window = 64,
    dead_feature_window=64,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "mats-hugo",
    wandb_entity = None,
    wandb_log_frequency=20,
    
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 0,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
    
    from_pretrained_path = 'checkpoints/e2oev6hw/final_sparse_autoencoder_llava-hf/llava-v1.6-vicuna-7b-hf_-2_resid_12288.pt'
    )

# torch.cuda.empty_cache()
# sparse_autoencoder, model = vision_transformer_sae_runner(cfg)
# sparse_autoencoder.eval()

torch.cuda.empty_cache()
if cfg.from_pretrained_path is not None:
  model, sparse_autoencoder, activations_loader = ViTSparseAutoencoderSessionloader.load_session_from_pretrained(cfg.from_pretrained_path)
  cfg = sparse_autoencoder.cfg
else:
  loader = ViTSparseAutoencoderSessionloader(cfg)
  model, sparse_autoencoder, activations_loader = loader.load_session()
  
print("model, sparse_auroencode and activations loading finish!!!")


def zero_ablation(activations):
  activations[:,0,:] = torch.zeros_like(activations[:,0,:]).to(activations.device)
  return (activations,) # activations of size [batch, token, dimension]
    
def sae_hook(activations):
    activations[:,-1,:] =  activations[:,-1,:]*0.5  # sparse_autoencoder(activations[:,-1,:])[0]    #   activations[:,0,:] = sparse_autoencoder(activations[:,0,:])[0]
    return (activations,)

def logits_to_next_str(logits, model = model):
  assert logits.shape[0] == 1
  logits = logits[:,-1,:].squeeze()
  next_tok = logits.argmax(dim = -1).item()  # scalar
  return model.to_string(next_tok)

if "cuda" in str(sparse_autoencoder.cfg.device):
  torch.cuda.empty_cache()
sparse_autoencoder.eval()


# 测试两种情况下用generate来生成
# model_inputs = activations_loader.get_batch_of_images_and_labels()
# input_ids = model_inputs["input_ids"]
# attention_mask = model_inputs["attention_mask"]
# pixel_values = model_inputs["pixel_values"]
# image_sizes = model_inputs["image_sizes"]
# max_token = 100
# generated_ids = input_ids.clone()
# generated_tokens = model.model.generate(
#     input_ids=input_ids, 
#     attention_mask=attention_mask,
#     pixel_values=pixel_values,
#     image_sizes=image_sizes,
#     max_new_tokens=max_token
# )
# output_texts = model.processor.tokenizer.batch_decode(
#     generated_tokens[0], skip_special_tokens=True
# )
# print(output_texts)

# generated_tokens = model.model.generate(**model_inputs, do_sample=True, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2, max_new_tokens=max_token)
# output_texts = model.processor.tokenizer.batch_decode(
#     generated_tokens[0], skip_special_tokens=True
# )
# print(output_texts)


# # 原来模型生成的
# model_inputs = activations_loader.get_batch_of_images_and_labels()
# input_ids = model_inputs["input_ids"]
# attention_mask = model_inputs["attention_mask"]
# pixel_values = model_inputs["pixel_values"]
# image_sizes = model_inputs["image_sizes"]
# max_token = 100
# generated_ids = input_ids.clone()
  
# with torch.no_grad():
#   for _ in range(max_token):
#     print("qingli333")
#     outputs = model(
#         return_type='output',
#         input_ids=generated_ids,
#         attention_mask=attention_mask,
#         pixel_values=pixel_values,
#         image_sizes=image_sizes,
#     )

#     logits = outputs.logits[:, -1, :]  
#     next_token = torch.argmax(logits, dim=-1).unsqueeze(-1) 
#     generated_ids = torch.cat([generated_ids, next_token], dim=-1)
#     new_mask = torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
#     attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
#     torch.cuda.empty_cache()

# output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)    
# print(output_texts)



# 添加hooks
model_inputs = activations_loader.get_batch_of_images_and_labels()
input_ids = model_inputs["input_ids"]
attention_mask = model_inputs["attention_mask"]
pixel_values = model_inputs["pixel_values"]
image_sizes = model_inputs["image_sizes"]
max_token = 100
generated_ids = input_ids.clone()
sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)]  
with torch.no_grad():
  for _ in range(max_token):
    print("qingli333")
    outputs = model.run_with_hooks(
        sae_hooks,
        return_type='output',
        input_ids=generated_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
    )

    logits = outputs.logits[:, -1, :]  
    next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
    print(next_token) 
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    
    new_mask = torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
    attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
    torch.cuda.empty_cache()

output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(output_texts)




