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
from transformers import AutoProcessor, LlavaForConditionalGeneration
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

def plot_alignment_fig(cos_sims):
    example_fig = px.line(cos_sims.to('cpu'))
    example_fig.show() 

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()
    

def create_scatter_plot(data, x_label = "", y_label = "", title="", colour_label = ""):
    num_columns = data.size()[0]
    assert num_columns in [2,3], "data should be of size (2,n) to create a scatter plot"
    data = data.transpose(0,1)
    
    if num_columns ==2:
        assert colour_label == "", "You can't submit a colour label when no colour variable is submitted"
        # Convert the torch tensor to a pandas DataFrame
        df = pd.DataFrame(data.numpy(), columns=[x_label, y_label])
        # Create the scatter plot with marginal histograms
        fig = px.scatter(df, x=x_label, y=y_label,
                        marginal_x='histogram', marginal_y='histogram',
                        )
    else:
        # Convert the torch tensor to a pandas DataFrame
        df = pd.DataFrame(data.numpy(), columns=[x_label, y_label, colour_label])
        # Create the scatter plot with marginal histograms
        fig = px.scatter(df, x=x_label, y=y_label, color=colour_label,
                        marginal_x='histogram', marginal_y='histogram',
                        color_continuous_scale=px.colors.sequential.Bluered,  # Optional: specify color scale     Bluered
                        )
        
    # Show the plot
    fig.update_layout(title={
        'text': title,
        'x': 0.5,  # Centering the title
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'family': 'Arial',
            'size': 24,
            'color': '#333333'
        }})
    # fig.show()
    fig.write_image("aaa222.png")


def get_sae_activations(model_activaitons, sparse_autoencoder):
    hook_name = "hook_hidden_post"
    max_batch_size = sparse_autoencoder.cfg.max_batch_size_for_vit_forward_pass # Use this for the SAE too
    number_of_mini_batches = model_activaitons.size()[0] // max_batch_size
    remainder = model_activaitons.size()[0] % max_batch_size
    sae_activations = []
    for mini_batch in trange(number_of_mini_batches, desc = "Dashboard: obtaining sae activations"):
        sae_activations.append(sparse_autoencoder.run_with_cache(model_activaitons[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size])[1][hook_name])
    
    if remainder>0:
        sae_activations.append(sparse_autoencoder.run_with_cache(model_activaitons[-remainder:])[1][hook_name])
        
    sae_activations = torch.cat(sae_activations, dim = 0)
    sae_activations = sae_activations.to(sparse_autoencoder.cfg.device)
    return sae_activations   
    
original_model = False
sae_model = True
neuron_alighment = False
scatter_plots = False
sae_sparsity = False


### specific case study---sparse autoencoder model
if sae_model:
    
    # sae_path = "checkpoints/0ns2guf8/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/a290660cfffb2fa669e809a8b52298dc901d2069/model.pt"
    # sae_path = "checkpoints/gsy8f8zy/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/99d8212c2b7e2d9661e1de1dab3b64b3dbb4f9b0/100864_sae_model.pt"
    sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/257fad408af4e96fb532887018dd8520cacc0a9f/302592_sae_model.pt"
    loaded_object = torch.load(sae_path)
    cfg = loaded_object['cfg']
    state_dict = loaded_object['state_dict']
    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.load_state_dict(state_dict)
    sparse_autoencoder.eval()

    loader = ViTSparseAutoencoderSessionloader(cfg)
    model = loader.get_model(cfg.model_name)
    model.to(cfg.device)

    print("model and sparse_auroencode loading finish!!!")

    # image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # raw_image = Image.open(requests.get(image_file, stream=True).raw)
    
    image_file = "image1.jpg"
    raw_image = Image.open(image_file)
    
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
            ],
        },
    ]
    prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
    model_inputs = model.processor(images=raw_image, text=prompt, return_tensors='pt').to(cfg.device, torch.float16)
    prompt_tokens = model_inputs.input_ids
    answer = "These are cat."
    # model_inputs = model.processor(text=answer, return_tensors='pt').to(0, torch.float16)
    # answer_tokens = model_inputs.input_ids
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    pixel_values = model_inputs.pixel_values
    # image_sizes = model_inputs.image_sizes

    tokenizer = model.processor.tokenizer
    prompt_str_tokens = tokenizer.convert_ids_to_tokens(prompt_tokens[0]) 
    # answer_str_tokens = tokenizer.convert_ids_to_tokens(answer_tokens[0])

        
    max_token = 20
    generated_ids = input_ids.clone()
    
    activation_cache = []
    def sae_hook1(activations):
        global activation_cache
        print("before:", activations.shape)
        activation_cache.append(activations[:, -2, :].clone().detach())
        activations[:,-2,:] = sparse_autoencoder(activations[:,-2,:])[0]
        print("After:", activations.shape)
        return (activations,)
    
    def zero_ablation(activations):
        activations[:,-1,:] = torch.zeros_like(activations[:,-1,:]).to(activations.device)
        return (activations,)
    
    def sae_hook(activations):
        global activation_cache
        activation_cache.append(activations[:, -2, :].clone().detach())
        activations[:,-2,:] = activations[:,-2,:]   
        return (activations,)
    
    # def sae_hook1(activations, original_output=None):
    #     global activation_cache
    #     activation_cache.append(activations[:, -1, :].clone().detach())
    #     activations[:, -1, :] = sparse_autoencoder(activations[:, -1, :])[0]  

    #     # Ensure the model gets all expected outputs
    #     if original_output is not None:
    #         return (activations, *original_output[1:])  # Preserve other return values
    #     return (activations,)

    
    
    # n_forward_passes_since_fired = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
    # ghost_grad_neuron_mask = (n_forward_passes_since_fired > sparse_autoencoder.cfg.dead_feature_window).bool()
    sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook1, return_module_output=True)]
    zero_ablation_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, zero_ablation, return_module_output=True)] 
    # sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, lambda activations, original_output: sae_hook1(activations, original_output),
    # return_module_output=True)]

    new_tokens = []
    with torch.no_grad():
        for ele in range(max_token):
            print(f"Token: {ele}")
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
            new_tokens.append(next_token)
            new_mask = torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            torch.cuda.empty_cache()

    output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(output_texts)
    
    
    number_of_top_features = 5
    activations = torch.cat(activation_cache, dim = 0)
    sae_activations = get_sae_activations(activations, sparse_autoencoder)
    values, indices = topk(sae_activations, k = number_of_top_features, dim = 1)
    
    token_number = 0
    for row1, row2 in zip(indices, values):
        token = model.processor.tokenizer.decode(new_tokens[token_number][0])
        print(f"Token:{token}")
        print("Features Indices:")
        print(row1)
        print("Features Values:")
        print(row2)
        token_number += 1
    qingli = 3
    




if original_model:
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    
    # image_file = "image1.jpg"
    # raw_image = Image.open(image_file)
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    model_inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
    output = model.generate(**model_inputs, max_new_tokens=50, do_sample=False)

    print(processor.decode(output[0][2:], skip_special_tokens=True))




if sae_sparsity:
    sparsity_tensor = torch.load('dashboard/sae_sparsity.pt').to('cpu')
    sparsity_tensor = torch.log10(sparsity_tensor)
    fig = xp.histogram(sparsity_tensor)
    fig.write_image("histogram.png")



if neuron_alighment:
    
    example_neurons = [25081,25097,38764,10186,14061,22552,41781,774,886,2681]
    sae_path = 'checkpoints/0ns2guf8/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt'
    loaded_object = torch.load(sae_path)
    cfg = loaded_object['cfg']
    state_dict = loaded_object['state_dict']
    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.load_state_dict(state_dict)
    sparse_autoencoder.eval()
    loader = ViTSparseAutoencoderSessionloader(cfg)
    model = loader.get_model(cfg.model_name)
    model.to(cfg.device)

    mlp_out_weights = model.model.language_model.model.layers[cfg.block_layer].mlp.down_proj.weight.detach().transpose(0,1) # size [hidden_mlp_dimemsion, resid_dimension]
    penultimate_mlp_out_weights = model.model.language_model.model.layers[cfg.block_layer-2].mlp.down_proj.weight.detach().transpose(0,1)
    sae_weights = sparse_autoencoder.W_enc.detach() # size [resid_dimension, sae_dimension]
    sae_weights /= torch.norm(sae_weights, dim = 0, keepdim = True)
    mlp_out_weights /= torch.norm(mlp_out_weights, dim = 1, keepdim = True)
    penultimate_mlp_out_weights /= torch.norm(penultimate_mlp_out_weights, dim = 1, keepdim = True)
    sae_weights = sae_weights.to(torch.float16)
    cosine_similarities = mlp_out_weights @ sae_weights # size [hidden_mlp_dimemsion, sae_dimension]


    cosine_similarities =torch.abs(cosine_similarities)
    max_cosine_similarities = torch.max(cosine_similarities, 0).values.to('cpu') # size [sae_dimension]
    subset_max_cosine_similarities = max_cosine_similarities[example_neurons]
    print(subset_max_cosine_similarities)
    mean_max_cos_sim = max_cosine_similarities.mean()
    var_max_cos_sim = max_cosine_similarities.var()

    threshold = 0.18
    num_above_threshold = (max_cosine_similarities>threshold).sum()

    fig = px.histogram(max_cosine_similarities, title = "Histogram of max cosine similarities of SAE features with MLP out tensor.")
    # fig.update_xaxes(range=[0.07, 1])
    # fig.show()
    fig.write_image("aaa.png")


    random_weights = torch.randn(sae_weights.size(), device = sae_weights.device)
    random_weights /= torch.norm(random_weights, dim = 0, keepdim = True)
    random_weights = random_weights.to(torch.float16)
    cosine_similarities = torch.abs(mlp_out_weights @ random_weights) # size [hidden_mlp_dimemsion, sae_dimension]
    max_cosine_similarities = torch.max(cosine_similarities, 0).values.to('cpu') # size [sae_dimension]
    rand_mean_max_cos_sim = max_cosine_similarities.mean()
    rand_var_max_cos_sim = max_cosine_similarities.var()

    rand_fig = px.histogram(max_cosine_similarities, title = "Histogram of max cosine similarities of random vectors with MLP out tensor.")
    # rand_fig.update_xaxes(range=[0.07, 1])
    # rand_fig.show()
    fig.write_image("bbb.png")

    cosine_similarities = torch.abs(penultimate_mlp_out_weights @ random_weights) # size [hidden_mlp_dimemsion, sae_dimension]
    max_cosine_similarities = torch.max(cosine_similarities, 0).values.to('cpu') # size [sae_dimension]
    rand_mean_max_cos_sim = max_cosine_similarities.mean()
    rand_var_max_cos_sim = max_cosine_similarities.var()

    rand_fig = px.histogram(max_cosine_similarities, title = "Histogram of max cosine similarities of SAE out tensor with MLP out tensor in the layer before.")
    # rand_fig.update_xaxes(range=[0.07, 1])
    # rand_fig.show()
    fig.write_image("ccc.png")



if scatter_plots:
    ### 1
    expansion_factor = 32
    directory = "dashboard"  # "dashboard" 
    sparsity = torch.load(f'{directory}/sae_sparsity.pt').to('cpu') # size [n]
    max_activating_image_indices = torch.load(f'{directory}/max_activating_image_indices.pt').to('cpu').to(torch.int32)
    max_activating_image_values = torch.load(f'{directory}/max_activating_image_values.pt').to('cpu')  # size [n, num_max_act]
    max_activating_image_label_indices =torch.load(f'{directory}/max_activating_image_label_indices.pt').to('cpu').to(torch.int32)  # size [n, num_max_act]
    sae_mean_acts = torch.load(f'{directory}/sae_mean_acts.pt').to('cpu')  # size [n]
    sae_mean_acts = max_activating_image_values.mean(dim = -1)

    df = pd.DataFrame(torch.log10(sparsity).numpy(), columns=['Data'])

    fig = px.histogram(df, x='Data', title='Sparsity histogram', nbins = 200)
    fig.update_layout(
        xaxis_title="Log 10 sparsity",
        yaxis_title="Count"
    )
    # fig.write_image("aaa111.png")
    
    ### 2
    number_of_neurons = max_activating_image_values.size()[0]
    entropy_list = torch.zeros(number_of_neurons)

    for i in range(number_of_neurons):
        # Get unique labels and their indices for the current sample
        unique_labels, _ = max_activating_image_label_indices[i].unique(return_inverse=True)
        unique_labels = unique_labels[unique_labels != 949] # ignore label 949 = dataset[0]['label'] - the default label index
        if len(unique_labels)!=0:
            counts = 0
            for label in unique_labels:
                counts += (max_activating_image_label_indices[i] == label).sum()
            if counts<10:
                entropy_list[i] = -1 # discount as too few datapoints!
            else:
                # Sum probabilities based on these labels
                summed_probs = torch.zeros_like(unique_labels, dtype = max_activating_image_values.dtype)
                for j, label in enumerate(unique_labels):
                    summed_probs[j] = max_activating_image_values[i][max_activating_image_label_indices[i] == label].sum().item()
                # Calculate entropy for the summed probabilities
                summed_probs = summed_probs / summed_probs.sum()  # Normalize to make it a valid probability distribution
                entropy = -torch.sum(summed_probs * torch.log(summed_probs + 1e-9))  # small epsilon to avoid log(0)
                entropy_list[i] = entropy
        else:
            entropy_list[i] = -1
            
    mask = (torch.log10(sparsity)>-4)&(torch.log10(sae_mean_acts)>-0.7)&(entropy_list>-1)
    
    print(f'Number of interesting neurons: {mask.sum()}')
    flattened_max_activating_image_indices = max_activating_image_indices[max_activating_image_indices!=0].flatten()
    most_common_index, _ = torch.mode(flattened_max_activating_image_indices)
    most_common_index = int(most_common_index)

    # Sparsity against mean activations
    data = torch.stack([torch.log10(sparsity[(entropy_list>-1)]+1e-9), torch.log10(sae_mean_acts[(entropy_list>-1)]+1e-9),entropy_list[(entropy_list>-1)]], dim = 0)
    create_scatter_plot(data, x_label = "Log 10 sparsity", y_label = "Log 10 mean activation value", title=f"Expansion factor {expansion_factor}", colour_label="Label Entropy")


    ### 3
    flattened_max_activating_image_label_indices = max_activating_image_label_indices[(torch.log10(sparsity)>-3.7)&(torch.log10(sae_mean_acts)>-0.4)].flatten()
    flattened_max_activating_image_label_indices = flattened_max_activating_image_label_indices[flattened_max_activating_image_label_indices !=0 ]
    label_frequency = [0 for _ in range(999)]
    for i in range(999):
        label_frequency[i] = (flattened_max_activating_image_label_indices==i).sum()
        
    fig = px.histogram(label_frequency)
    fig.write_image("aaa333.png")

    fig = px.line(label_frequency)
    fig.write_image("aaa444.png")


    ### 4
    sae_path = f"checkpoints/0ns2guf8/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt"
    loaded_object = torch.load(sae_path)
    cfg = loaded_object['cfg']
    state_dict = loaded_object['state_dict']
    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.load_state_dict(state_dict)
    sparse_autoencoder.eval()
    loader = ViTSparseAutoencoderSessionloader(cfg)
    model = loader.get_model(cfg.model_name)
    model.to(cfg.device)
    dataset = load_dataset(cfg.dataset_path, split="train")
    dataset = dataset.shuffle(seed = 1)
    image = dataset[most_common_index]['image']
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]
    
    conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"}, 
                {"type": "image"},
                ],
            },
        ] 
    prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = model.processor(images=[image], text = [prompt], return_tensors="pt", padding = True).to(cfg.device)
    model_activations = model.run_with_cache(
        list_of_hook_locations,
        **inputs,
    )[1][(block_layer, module_name)]
    model_activations = model_activations[:,-1,:]
    _, feature_acts, _, _, _, _ = sparse_autoencoder(model_activations)
    feature_acts = feature_acts.to('cpu')
    print(f'Most common image has index: {most_common_index}')
    print(f'Number of sae features that fired: {(feature_acts>0).sum()}')
    print(f'Percentage of neurons that fired: {(feature_acts>0).sum()/(expansion_factor*1024):.2%}')
    mean_log_sparsity = torch.log10(sparsity[feature_acts.squeeze()>0]).mean()
    print(f'Mean log 10 sparsity of those that fired: {mean_log_sparsity}')
        







