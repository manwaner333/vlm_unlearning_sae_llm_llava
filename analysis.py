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
from transformers import AutoProcessor, LlavaForConditionalGeneration
from plotly import express as xp
import torch
import plotly.io as pio
from typing import Union, List, Optional
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter


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


def conversation_form(key):
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": key},  # "What's in the picture?"
            {"type": "image"},
            ],
        },
    ]
    return conversation


# def get_model_activations(model, inputs, cfg):
#     module_name = cfg.module_name
#     block_layer = cfg.block_layer
#     list_of_hook_locations = [(block_layer, module_name)]

#     activations = model.run_with_cache(
#         list_of_hook_locations,
#         **inputs,
#     )[1][(block_layer, module_name)]
    
#     activations = activations[:,-7,:]

#     return activations

def get_model_activations(model, inputs, cfg):
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]

    activations = model.run_with_cache(
        list_of_hook_locations,
        **inputs,
    )[1][(block_layer, module_name)]
    
    activations = activations[:,4:-5,:]
    # activations = activations[:,575:,:]

    return activations


def get_all_model_activations(model, images, conversations, cfg):
    max_batch_size = cfg.max_batch_size_for_vit_forward_pass
    number_of_mini_batches = len(images) // max_batch_size
    remainder = len(images) % max_batch_size
    sae_batches = []
    for mini_batch in trange(number_of_mini_batches, desc = "Dashboard: forward pass images through ViT"):
        image_batch = images[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size]
        conversation_batch = conversations[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size]
        # inputs = model.processor(images=image_batch, text = "", return_tensors="pt", padding = True).to(model.model.device)
        # conversation = [
        #     {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": "What is shown in this image?"},
        #         {"type": "image"},
        #         ],
        #     },
        # ]
        # prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
        batch_of_prompts = []
        for ele in conversation_batch:
            batch_of_prompts.append(model.processor.apply_chat_template(ele, add_generation_prompt=True))
        
        inputs = model.processor(images=image_batch, text=batch_of_prompts, padding=True, return_tensors="pt").to(model.model.device)
        sae_batches.append(get_model_activations(model, inputs, cfg))
    
    if remainder>0:
        image_batch = images[-remainder:]
        conversation_batch = conversations[-remainder:]
        # inputs = model.processor(images=image_batch, text = "", return_tensors="pt", padding = True).to(model.model.device)
        # conversation = [
        #     {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": "What is shown in this image?"},
        #         {"type": "image"},
        #         ],
        #     },
        # ]
        # prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
        batch_of_prompts = []
        for ele in conversation_batch:
            batch_of_prompts.append(model.processor.apply_chat_template(ele, add_generation_prompt=True))
            
        inputs = model.processor(images=image_batch, text=batch_of_prompts, padding=True, return_tensors="pt").to(model.model.device)
        sae_batches.append(get_model_activations(model, inputs, cfg))
        
    sae_batches = torch.cat(sae_batches, dim = 0)
    sae_batches = sae_batches.to(cfg.device)
    return sae_batches


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
  
original_model = False
sae_model = False
neuron_alighment = False
scatter_plots = False
sae_sparsity = False
focus_features = False
focus_images = False


### specific case study---sparse autoencoder model
if sae_model:
    
    # sae_path = "checkpoints/0ns2guf8/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/a290660cfffb2fa669e809a8b52298dc901d2069/model.pt"
    # sae_path = "checkpoints/gsy8f8zy/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/99d8212c2b7e2d9661e1de1dab3b64b3dbb4f9b0/100864_sae_model.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/257fad408af4e96fb532887018dd8520cacc0a9f/302592_sae_model.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/3c468d5fd49342de8f38dcf7a4eecaeb4e8b6ec6/100864_sae_image_model.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/85e02735659e04a6e74651b74784fd1d19168b18/100864_sae_image_model_activations_7.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/91251c64c0a50806a46a97e16f4b77ac65e37a04/201728_sae_image_model_activations_7.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/11e422e9a6b886457af1f53b095fdbc401d68233/302592_sae_image_model_activations_7.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/9ae094c2e23727d1c77d05d46f419d2b1e2e6aef/605184_sae_image_model_activations_7.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/aa9c6eb62ded51020e8c5c34182602af353d9d77/1210112_sae_image_model_activations_7.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/3cab4c8243f1f0954b74f45f3a7ba64ffaba073b/1714176_sae_image_model_activations_7.pt"
    sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/e44861c762f4a32084ac448f31cd7264800610df/2621440_sae_image_model_activations_7.pt"
    
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
    
    # image_file = "image2.png"
    image_file = "dashboard_2621440/images/image_2.png"
    raw_image = Image.open(image_file)
    
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe this picture."},   # "What are these?"
            {"type": "image"},
            ],
        },
    ]
    prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
    model_inputs = model.processor(images=raw_image, text=prompt, return_tensors='pt').to(cfg.device, torch.float16)
    prompt_tokens = model_inputs.input_ids
    # answer = "These are cat."
    # model_inputs = model.processor(text=answer, return_tensors='pt').to(0, torch.float16)
    # answer_tokens = model_inputs.input_ids
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    pixel_values = model_inputs.pixel_values
    # image_sizes = model_inputs.image_sizes

    tokenizer = model.processor.tokenizer
    prompt_str_tokens = tokenizer.convert_ids_to_tokens(prompt_tokens[0]) 
    # answer_str_tokens = tokenizer.convert_ids_to_tokens(answer_tokens[0])

        
    max_token = 1024
    generated_ids = input_ids.clone()
    
    activation_cache = []
    # def sae_hook1(activations):
    #     global activation_cache
    #     print("before:", activations.shape)
    #     activation_cache.append(activations[:, -2, :].clone().detach())
    #     activations[:,-2,:] = sparse_autoencoder(activations[:,-2,:])[0]
    #     print("After:", activations.shape)
    #     return (activations,)
    
    # def zero_ablation(activations):
    #     activations[:,-1,:] = torch.zeros_like(activations[:,-1,:]).to(activations.device)
    #     return (activations,)
    
    def sae_hook(activations):
        global activation_cache
        # LLMs
        # activation_cache.append(activations[:, -1, :].clone().detach())
        # activations[:,-1,:] = activations[:,-1,:] 
        # ViTs
        # activation_cache.append(activations[:, -1, :].clone().detach())
        activations[:,0:576,:] = sparse_autoencoder(activations[:,0:576,:])[0]    
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
    sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)]
    # zero_ablation_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, zero_ablation, return_module_output=True)] 
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
    
    
    # number_of_top_features = 5
    # activations = torch.cat(activation_cache, dim = 0)
    # sae_activations = get_sae_activations(activations, sparse_autoencoder)
    # values, indices = topk(sae_activations, k = number_of_top_features, dim = 1)
    
    # token_number = 0
    # for row1, row2 in zip(indices, values):
    #     token = model.processor.tokenizer.decode(new_tokens[token_number][0])
    #     print(f"Token:{token}")
    #     print("Features Indices:")
    #     print(row1)
    #     print("Features Values:")
    #     print(row2)
    #     token_number += 1
    # qingli = 3
    




if original_model:
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # raw_image = Image.open(requests.get(image_file, stream=True).raw)
    
    image_file = "dashboard_2621440/images/image_2.png"  # "image2.png"
    raw_image = Image.open(image_file)
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe this picture."},  # "What are these?"
            {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    model_inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
    output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)

    print(processor.decode(output[0][2:], skip_special_tokens=True))


if sae_sparsity:
    # SAE 一共65536个特征。 统计的时候使用了524288张图片。 每一个特征在这524288张图片上非零的数量/524288, 就是每一个特征对应的值， 记为sae sparisity。 因此该表统计了65536个特征的状况。
    sparsity_tensor = torch.load('dashboard/sae_sparsity.pt').to('cpu')
    # sparsity_tensor = torch.clamp(sparsity_tensor, min=1e-10)
    sparsity_tensor = torch.log10(sparsity_tensor)
    fig = xp.histogram(sparsity_tensor)
    
    fig.update_traces(marker=dict(color="#4169E1"))
    
    fig.update_xaxes(
        title_text=r'$\LARGE \log_{10}(\text{FS})$',  # X-axis label
        # title_font_size=30,                   # X-axis label font size
        # tickfont_size=26,                      # X-axis tick font size
        title_font=dict(size=28, color='black'),
        tickfont=dict(size=26, color='black'),
        # tickformat=".e"
        showline=True,       # 显示 X 轴线
        linewidth=1,         # 线宽
        linecolor='black',    # 线颜色
        mirror=True
    )
    fig.update_yaxes(
        title_text='Number of Features',               # Y-axis label
        # title_font_size=30,                   # Y-axis label font size
        # tickfont_size=26,
        title_font=dict(size=28, color='black'),
        tickfont=dict(size=26, color='black'),
        showline=True,       # 显示 X 轴线
        linewidth=1,         # 线宽
        linecolor='black',    # 线颜色
        mirror=True
    )
    
    fig.update_layout(
    # paper_bgcolor='lightgrey', 
    # plot_bgcolor='white' 
    plot_bgcolor='#f7f7f7' 
    )

    # Customize the legend (if applicable)
    # fig.update_layout(
    #     legend_title_text='Legend Title',     # Legend title
    #     legend_font_size=14,                  # Legend font size
    #     legend=dict(
    #         x=0.8,                            # Legend x position (0 to 1, relative to the plot)
    #         y=0.9,                            # Legend y position (0 to 1, relative to the plot)
    #         bgcolor='rgba(255, 255, 255, 0.5)' # Legend background color (optional)
    #     )
    # )
    
    fig.update_layout(margin=dict(l=0, r=5, t=5, b=0))
    fig.update_layout(showlegend=False, 
                      legend=dict(font=dict(size=13)))
    
    
    # fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.write_image("histogram_update.png", scale=2)
    
    # sparsity_tensor[sparsity_tensor == -float('inf')] = torch.min(sparsity_tensor[sparsity_tensor != -float('inf')])
    
    # sparsity_tensor = sparsity_tensor.numpy()
    # hist, bin_edges = np.histogram(sparsity_tensor, bins=10)
    # plt.hist(bin_edges[:-1], bins=bin_edges, weights=hist, edgecolor='black', label='Histogram')

    # # Set x and y axis labels with custom font size
    # plt.xlabel('Log10 of Sparsity Value', fontsize=14)  # Customize x-axis label
    # plt.ylabel('Frequency', fontsize=14)  # Customize y-axis label

    # # Set title with custom font size
    # plt.title('Histogram of Sparsity Values', fontsize=16)

    # # Customize tick label sizes
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    # # Add a legend with custom font size
    # plt.legend(fontsize=12)

    # # Save the plot if needed
    # plt.savefig('sparsity_histogram.png', dpi=300)

    # # Optionally, close the plot to avoid overlapping with future plots
    # plt.close()
        
    



if neuron_alighment:
    
    example_neurons = [25081,25097,38764,10186,14061,22552,41781,774,886,2681]
    # sae_path = 'checkpoints/0ns2guf8/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt'
    sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/e44861c762f4a32084ac448f31cd7264800610df/2621440_sae_image_model_activations_7.pt"
    loaded_object = torch.load(sae_path)
    cfg = loaded_object['cfg']
    state_dict = loaded_object['state_dict']
    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.load_state_dict(state_dict)
    sparse_autoencoder.eval()
    loader = ViTSparseAutoencoderSessionloader(cfg)
    model = loader.get_model(cfg.model_name)
    model.to(cfg.device)

    # mlp_out_weights = model.model.language_model.model.layers[cfg.block_layer].mlp.down_proj.weight.detach().transpose(0,1) # size [hidden_mlp_dimemsion, resid_dimension]
    # penultimate_mlp_out_weights = model.model.language_model.model.layers[cfg.block_layer-2].mlp.down_proj.weight.detach().transpose(0,1)
    mlp_out_weights = model.model.vision_tower.vision_model.encoder.layers[cfg.block_layer].mlp.down_proj.weight.detach().transpose(0,1)
    penultimate_mlp_out_weights = model.model.vision_tower.vision_model.encoder.layers[cfg.block_layer-2].mlp.down_proj.weight.detach().transpose(0,1)
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
    # ### 1
    expansion_factor = 64
    directory = "dashboard"  # "dashboard" 
    sparsity = torch.load(f'{directory}/sae_sparsity.pt').to('cpu') # size [n]
    max_activating_image_indices = torch.load(f'{directory}/max_activating_image_indices.pt').to('cpu').to(torch.int32)
    max_activating_image_values = torch.load(f'{directory}/max_activating_image_values.pt').to('cpu')  # size [n, num_max_act]
    max_activating_image_label_indices =torch.load(f'{directory}/max_activating_image_label_indices.pt').to('cpu').to(torch.int32)  # size [n, num_max_act]
    sae_mean_acts = torch.load(f'{directory}/sae_mean_acts.pt').to('cpu')  # size [n]
    sae_mean_acts = max_activating_image_values.mean(dim = -1)

    # df = pd.DataFrame(torch.log10(sparsity).numpy(), columns=['Data'])

    # fig = px.histogram(df, x='Data', title='Sparsity histogram', nbins = 200)
    # fig.update_layout(
    #     xaxis_title="Log 10 sparsity",
    #     yaxis_title="Count"
    # )
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
    # fig.write_image("aaa333.png")

    fig = px.line(label_frequency)
    # fig.write_image("aaa444.png")
    
    ### 4
    sae_mean_acts_tensor = torch.log10(sae_mean_acts)
    fig = xp.histogram(sae_mean_acts_tensor)
    fig.update_traces(marker=dict(color="#4169E1"))
    
    fig.update_xaxes(
        title_text=r'$\LARGE \log_{10}(\text{FMAV})$',      # X-axis label
        # title_font_size=32,                   # X-axis label font size
        # tickfont_size=28                      # X-axis tick font size
        title_font=dict(size=28, color='black'),
        tickfont=dict(size=26, color='black'),
        showline=True,       # 显示 X 轴线
        linewidth=1,         # 线宽
        linecolor='black',    # 线颜色
        mirror=True
        
    )
    # fig.update_xaxes(
    #     title={
    #         'text': r'$\log_{10}(\text{FMAV})$',  # X 轴标题
    #         'font': {'size': 32}                  # 标题字体大小
    #     },
    #     tickfont={'size': 28}                      # 刻度字体大小
    # )
    fig.update_yaxes(
        title_text='Number of Features',                 # Y-axis label
        # title_font_size=32,                   # Y-axis label font size
        # tickfont_size=28                      # Y-axis tick font size
        title_font=dict(size=28, color='black'),
        tickfont=dict(size=26, color='black'),
        showline=True,       # 显示 X 轴线
        linewidth=1,         # 线宽
        linecolor='black',    # 线颜色
        mirror=True
    )
    
    fig.update_layout(
        plot_bgcolor='#f7f7f7'
        # paper_bgcolor='lightgrey', 
        # plot_bgcolor='white'  
    )
    # Customize the legend (if applicable)
    # fig.update_layout(
    #     legend_title_text='Legend Title',     # Legend title
    #     legend_font_size=14,                  # Legend font size
    #     legend=dict(
    #         x=0.8,                            # Legend x position (0 to 1, relative to the plot)
    #         y=0.9,                            # Legend y position (0 to 1, relative to the plot)
    #         bgcolor='rgba(255, 255, 255, 0.5)' # Legend background color (optional)
    #     )
    # )
    
    fig.update_layout(margin=dict(l=0, r=5, t=5, b=0))
    fig.update_layout(showlegend=False, legend=dict(font=dict(size=13)))
    fig.write_image("sae_mean_acts_update.png", scale=2)
    


    ### 5
    # # sae_path = f"checkpoints/0ns2guf8/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt"
    # # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/11e422e9a6b886457af1f53b095fdbc401d68233/302592_sae_image_model_activations_7.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/e44861c762f4a32084ac448f31cd7264800610df/2621440_sae_image_model_activations_7.pt"
    # loaded_object = torch.load(sae_path)
    # cfg = loaded_object['cfg']
    # state_dict = loaded_object['state_dict']
    # sparse_autoencoder = SparseAutoencoder(cfg)
    # sparse_autoencoder.load_state_dict(state_dict)
    # sparse_autoencoder.eval()
    # loader = ViTSparseAutoencoderSessionloader(cfg)
    # model = loader.get_model(cfg.model_name)
    # model.to(cfg.device)
    # dataset = load_dataset(cfg.dataset_path, split="train")
    # dataset = dataset.shuffle(seed = 1)
    # image = dataset[most_common_index]['image']
    # module_name = cfg.module_name
    # block_layer = cfg.block_layer
    # list_of_hook_locations = [(block_layer, module_name)]
    
    # conversation = [
    #         {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "What is shown in this image?"}, 
    #             {"type": "image"},
    #             ],
    #         },
    #     ] 
    # prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
    # inputs = model.processor(images=[image], text = [prompt], return_tensors="pt", padding = True).to(cfg.device)
    # model_activations = model.run_with_cache(
    #     list_of_hook_locations,
    #     **inputs,
    # )[1][(block_layer, module_name)]
    # model_activations = model_activations[:,-7,:]
    # _, feature_acts, _, _, _, _ = sparse_autoencoder(model_activations)
    # feature_acts = feature_acts.to('cpu')
    # print(f'Most common image has index: {most_common_index}')
    # print(f'Number of sae features that fired: {(feature_acts>0).sum()}')
    # print(f'Percentage of neurons that fired: {(feature_acts>0).sum()/(expansion_factor*1024):.2%}')
    # mean_log_sparsity = torch.log10(sparsity[feature_acts.squeeze()>0]).mean()
    # print(f'Mean log 10 sparsity of those that fired: {mean_log_sparsity}')
        


if focus_features:

    seed = 1
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/11e422e9a6b886457af1f53b095fdbc401d68233/302592_sae_image_model_activations_7.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/9ae094c2e23727d1c77d05d46f419d2b1e2e6aef/605184_sae_image_model_activations_7.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/aa9c6eb62ded51020e8c5c34182602af353d9d77/1210112_sae_image_model_activations_7.pt"
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/3cab4c8243f1f0954b74f45f3a7ba64ffaba073b/1714176_sae_image_model_activations_7.pt"
    sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/e44861c762f4a32084ac448f31cd7264800610df/2621440_sae_image_model_activations_7.pt"
    loaded_object = torch.load(sae_path)
    cfg = loaded_object['cfg']
    state_dict = loaded_object['state_dict']
    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.load_state_dict(state_dict)
    sparse_autoencoder.eval()

    dataset = load_dataset(sparse_autoencoder.cfg.dataset_path, split="train")
    image_label = 'label' 
    image_key = 'image'
    dataset = dataset.shuffle(seed = seed)
    directory = "dashboard"
        
    max_activating_image_indices = torch.load(f'{directory}/max_activating_image_indices.pt').to('cpu').to(torch.int32)
    max_activating_image_values = torch.load(f'{directory}/max_activating_image_values.pt').to('cpu') 

    # 找出激活着比较高的特征， 以及这些特征对应的比较强的图片
    # num_neurons = 20
    # row_sums = max_activating_image_values.sum(dim=1, keepdim=True)
    # top20_indices = torch.topk(row_sums.squeeze(), 30).indices
    # print(top20_indices)

    # neuron_list = [43208, 17533,  2950, 34679, 49570,  194, 35429, 28796, 42148,  7401, 14129, 63464, 24866, 19612,  8645, 13087, 17839, 52874, 45455,  6295] 
    # neuron_list = [43208, 17533,  2950, 34679, 49570,  194, 35429, 28796, 42148]   
    # neuron_list = [18048, 15239, 40921, 16830, 20003,  9512,  9537, 18422,   837, 10637, 17803,  4624, 40132, 34198, 18922, 16183, 56256, 52172, 11539, 58812]
    # neuron_list = [18048, 15239, 40921, 16830, 20003,  9512,  9537, 18422,   837]
    # neuron_list = [18048, 40921, 15239,  9512, 15756, 12468, 16830,  7802, 14260, 60111, 9537, 56821, 10506,  1197, 20003, 52355, 24553, 15821, 50541, 11382]
    # neuron_list = [18048, 40921, 15239,  9512, 15756, 12468, 16830,  7802, 14260, 60111, 9537]

    # neuron_list = [42524, 30191,  1102,  7175, 16147, 31263, 43934, 30087, 13127, 29133, 8020, 18048, 18029, 24195, 15821, 54816,  9512, 58213, 10251,  3402]   # 
    neuron_list = [45419, 16148, 24553, 33397, 15756, 34674, 52355, 14316,  4882, 42623]
    assert max_activating_image_values.size() == max_activating_image_indices.size(), "size of max activating image indices doesn't match the size of max activing values."
    number_of_neurons, number_of_max_activating_examples = max_activating_image_values.size()
    # for neuron in trange(number_of_neurons):
    for neuron in neuron_list:
        print("qingli")
        neuron_dead = True
        for max_activating_image in range(number_of_max_activating_examples):
            if max_activating_image_values[neuron, max_activating_image].item()>0:
                if neuron_dead:
                    if not os.path.exists(f"{directory}/{neuron}"):
                        os.makedirs(f"{directory}/{neuron}")
                    neuron_dead = False
                image = dataset[int(max_activating_image_indices[neuron, max_activating_image].item())][image_key]
                image.save(f"{directory}/{neuron}/{max_activating_image}_{int(max_activating_image_indices[neuron, max_activating_image].item())}_{max_activating_image_values[neuron, max_activating_image].item():.4g}.png", "PNG")
                
                



if focus_images:
    
    # sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/e44861c762f4a32084ac448f31cd7264800610df/2621440_sae_image_model_activations_7.pt"
    # loaded_object = torch.load(sae_path)
    # cfg = loaded_object['cfg']
    # state_dict = loaded_object['state_dict']

    # sparse_autoencoder = SparseAutoencoder(cfg)
    # sparse_autoencoder.load_state_dict(state_dict)
    # sparse_autoencoder.eval()

    # loader = ViTSparseAutoencoderSessionloader(cfg)

    # model = loader.get_model(cfg.model_name)
    # model.to(cfg.device)

    # torch.cuda.empty_cache()
    # dataset_all = load_dataset(sparse_autoencoder.cfg.dataset_path, split="train")
    # label = 0
    # dataset_5000 = dataset_all.select(range(0, 5000))
    # # label = 200   # dog
    # # dataset_5000 = dataset_all.select(range(255000, 265000))
    # dataset = dataset_5000.filter(lambda example: example['label'] == label)
    # print(f"Total data quantity: {len(dataset)}")


    # if sparse_autoencoder.cfg.dataset_path=="cifar100": # Need to put this in the cfg
    #     image_key = 'img'
    # else:
    #     image_key = 'image'

    # image_label = 'label' 
    # dataset = dataset.shuffle(seed = seed)
    # directory = "dashboard_2621440"


    # # 保存某张相关图片
    # # first_image = dataset[3][image_key]
    # # if isinstance(first_image, Image.Image):
    # #     first_image.save("first_image.png")  
    # #     print("Image saved as first_image.png")
    # # elif isinstance(first_image, dict) and "bytes" in first_image:
    # #     img = Image.open(io.BytesIO(first_image["bytes"]))
    # #     img.save("first_image.png")
    # #     print("Image saved as first_image.png")
    # # else:
    # #     print("Unsupported image format:", type(first_image))



    # number_of_images_processed = 0
    # max_number_of_images_per_iteration = len(dataset)
    # while number_of_images_processed < len(dataset):
    #     torch.cuda.empty_cache()
    #     try:
    #         images = dataset[number_of_images_processed:number_of_images_processed + max_number_of_images_per_iteration][image_key]
    #         labels = dataset[number_of_images_processed:number_of_images_processed + max_number_of_images_per_iteration][image_label]
    #         conversations = [conversation_form(str(ele)) for ele in labels]
    #     except StopIteration:
    #         print('All of the images in the dataset have been processed!')
    #         break
        
    #     model_activations = get_all_model_activations(model, images, conversations, sparse_autoencoder.cfg) 
    #     sae_activations = get_sae_activations(model_activations, sparse_autoencoder).transpose(0,1) 
    #     torch.save(sae_activations, f'{directory}/sae_activations_{label}.pt')
    #     number_of_images_processed += max_number_of_images_per_iteration




    directory = "dashboard_2621440"
        
    sae_activations_0 = torch.load(f'{directory}/sae_activations_fish.pt').to('cpu')
    sae_activations_200 = torch.load(f'{directory}/sae_activations_dog.pt').to('cpu') 

    sum_0 = sae_activations_0.mean(dim=1, keepdim=True)
    sum_200 = sae_activations_200.mean(dim=1, keepdim=True)

    epsilon = torch.finfo(sum_0.dtype).eps
    total_sum_0 = sum_0.sum() + epsilon 
    total_sum_200 = sum_200.sum() + epsilon

    ratio_0 = sum_0 / total_sum_0 
    ratio_200 = sum_200 / total_sum_200 


    # diff_ratio = ratio_0 - ratio_200
    diff_ratio = ratio_200 - ratio_0
    values, indices = torch.topk(diff_ratio, 10, dim=0)

    print("value of epsion:", epsilon)
    print("total_sum:", total_sum_0)
    print("a shape:", sum_0.shape)        
    print("ratio shape:", ratio_0.shape)
    print("Top 10 indices:", indices.squeeze().tolist())
    print("Top 10 values:", values.squeeze().tolist())
    
    
    




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



#####  尝试找一下和name 相关的特征

dataset_path = "MLLMMU/MLLMU-Bench"
dataset = load_dataset(dataset_path, "Full_Set")['train']
# name_list = [
#     "Albert Einstein", "Isaac Newton", "Marie Curie", "Galileo Galilei", "Charles Darwin",
#     "Stephen Hawking", "Nikola Tesla", "Leonardo da Vinci", "Aristotle", "Plato",
#     "Confucius", "Socrates", "Immanuel Kant", "Karl Marx", "Sigmund Freud",
#     "William Shakespeare", "J.K. Rowling", "Leo Tolstoy", "Mark Twain", "Jane Austen",
#     "Ernest Hemingway", "George Orwell", "Victor Hugo", "Fyodor Dostoevsky", "Homer",
#     "Mozart", "Beethoven", "Bach", "Frederic Chopin", "Tchaikovsky",
#     "Elvis Presley", "Michael Jackson", "The Beatles", "Madonna", "Bob Dylan",
#     "Barack Obama", "Donald Trump", "Joe Biden", "Hillary Clinton", "George W. Bush",
#     "Abraham Lincoln", "George Washington", "Franklin D. Roosevelt", "Theodore Roosevelt", "John F. Kennedy",
#     "Napoleon Bonaparte", "Winston Churchill", "Nelson Mandela", "Mahatma Gandhi", "Martin Luther King Jr.",
#     "Angela Merkel", "Vladimir Putin", "Xi Jinping", "Kim Jong Un", "Emmanuel Macron",
#     "Jeff Bezos", "Elon Musk", "Bill Gates", "Mark Zuckerberg", "Steve Jobs",
#     "Oprah Winfrey", "Ellen DeGeneres", "Tom Hanks", "Leonardo DiCaprio", "Brad Pitt",
#     "Angelina Jolie", "Scarlett Johansson", "Jennifer Lawrence", "Johnny Depp", "Robert Downey Jr.",
#     "Cristiano Ronaldo", "Lionel Messi", "Neymar", "Pelé", "Diego Maradona",
#     "Roger Federer", "Serena Williams", "Michael Jordan", "LeBron James", "Kobe Bryant",
#     "Tiger Woods", "Usain Bolt", "Muhammad Ali", "Mike Tyson", "Rafael Nadal",
#     "Greta Thunberg", "Malala Yousafzai", "Aung San Suu Kyi", "Desmond Tutu", "Dalai Lama",
#     "Pope Francis", "Mother Teresa", "Queen Elizabeth II", "Princess Diana", "Prince William",
#     "Taylor Swift", "Beyoncé", "Rihanna", "Lady Gaga", "Adele",
#     "Drake", "Kanye West", "Justin Bieber", "Shakira", "Ed Sheeran"
# ]

feature_indices = []
# for index in range(0, len(dataset)):
for index in range(0, len(dataset)):
    image = dataset[index]['image']
    biography = dataset[index]['biography']
    name = "Donald_Trump"
    # name = json.loads(biography)['Name']
    # name = name_list[index]
    # name = json.loads(biography)['Employment']
    # if 'Height' in json.loads(biography).keys():
    #     name = json.loads(biography)['Height']
    # elif 'Heights' in json.loads(biography).keys():
    #     name = json.loads(biography)['Heights']
    # else:
    #     print("missing")
    #     continue
    print(f"name: {name}")
    conversation = conversation_only_text_form(name)
    prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    token_ids = model.processor.tokenizer(prompt)['input_ids']
    text_list = [model.processor.tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
    print(text_list[3])
    print(text_list[-5])
    
    inputs = model.processor(text=prompt, return_tensors='pt').to(0, torch.float16)
    model_activations = get_model_activations(model, inputs, sparse_autoencoder.cfg)

    hook_name = "hook_hidden_post"
    sae_activations = sparse_autoencoder.run_with_cache(model_activations)[1][hook_name][0].transpose(0,1)
    sae_activations_sum = sae_activations.sum(dim = 1)
    values, indices = topk(sae_activations_sum, k = 20)
    feature_indices.append(indices.tolist())

flattened = [item for row in feature_indices for item in row]
counter = Counter(flattened)

with open(f'dataset/name_features_info_update.json', 'a') as file:
# with open(f'dataset/employment_features_info.json', 'a') as file:
# with open(f'dataset/height_features_info.json', 'a') as file:
    for item, count in counter.most_common():
        print(f"{item}: {count}", file=file)



# 找出激活值最大的一些特征。
# max_activating_image_indices = torch.load(f'dashboard/max_activating_image_indices.pt').to('cpu').to(torch.int32)
# max_activating_image_values = torch.load(f'dashboard/max_activating_image_values.pt').to('cpu')
# row_mean = max_activating_image_values.mean(dim=1)
# top10_values, top10_indices = torch.topk(row_mean, k=100)
# print(top10_indices)
# print(top10_values)



##### 从一批数据中， 选出某个特征对应的激活值最大的一些token
# max_activating_image_indices = torch.load(f'dashboard/max_activating_image_indices.pt').to('cpu').to(torch.int32)
# max_activating_image_values = torch.load(f'dashboard/max_activating_image_values.pt').to('cpu')
# tokens_file = "dashboard/dataset_output_tokens.json"
# feature_tokens_indices = max_activating_image_indices[5901, :].detach().cpu().numpy()

# tokens_data = []
# with open(tokens_file, "r", encoding="utf-8") as f:
#     for line in f:
#         value = json.loads(line)
#         tokens_data.extend(value)   

# feature_tokens_ids = []
# for token_indice in feature_tokens_indices:
#     quotient, remainder = divmod((token_indice+1), 614)
#     quotient_adj = quotient - 1
#     remainder_adj = remainder - 1
#     feature_tokens_ids.append(tokens_data[quotient_adj][remainder_adj])
# print(feature_tokens_ids)

# text = model.processor.tokenizer.decode(feature_tokens_ids)
# print(f"text: {text}")



# #####  给出一句话， 提取里面某个特征在各个token上对应的激活值

# dataset_path = "MLLMMU/MLLMU-Bench"
# forget_dataset = load_dataset(dataset_path, "forget_10")['train']
# retain_dataset = load_dataset(dataset_path, "retain_90")['train']

# forget_index = 0
# forget_image = forget_dataset[forget_index]['image']
# forget_biography = forget_dataset[forget_index]['biography']
# forget_conversation = conversation_form(forget_biography)
# forget_prompt = model.processor.apply_chat_template(forget_conversation, add_generation_prompt=True)
# forget_inputs = model.processor(images=forget_image, text=forget_prompt, return_tensors='pt').to(0, torch.float16)
# forget_model_activations = get_model_activations(model, forget_inputs, sparse_autoencoder.cfg)

# forget_token_ids = model.processor.tokenizer(forget_prompt)['input_ids']
# forget_text_list = [model.processor.tokenizer.decode([tid], skip_special_tokens=False) for tid in forget_token_ids]

# hook_name = "hook_hidden_post"
# forget_sae_activations = sparse_autoencoder.run_with_cache(forget_model_activations)[1][hook_name]
# feature_index = 39356
# feature_sae_activations = forget_sae_activations[0].transpose(0,1)[feature_index, :].detach().cpu().numpy()


# # 示例数据
# list1 = text_list
# list2 = feature_sae_activations

# # 创建一个颜色映射 - 使用蓝色调
# cmap = plt.cm.Blues

# # 计算字符串的总长度和位置
# positions = []
# current_pos = 0
# for word in list1:
#     positions.append((current_pos, current_pos + len(word)))
#     current_pos += len(word) + 1  # +1 为了单词之间的空格

# # 归一化值以便于颜色映射
# min_val = min(list2)
# max_val = max(list2)
# normalized_values = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0.5 for val in list2]

# # 设置图形
# fig, ax = plt.subplots(figsize=(40, 1.5))
# ax.set_xlim(0, current_pos)
# ax.set_ylim(0, 1)
# ax.axis('off')  # 隐藏坐标轴

# # 绘制带颜色的矩形和文本
# for i, (word, val, norm_val) in enumerate(zip(list1, list2, normalized_values)):
#     start, end = positions[i]
#     width = end - start
    
#     # 创建颜色矩形
#     color = cmap(norm_val)
#     rect = patches.Rectangle((start, 0), width, 1, 
#                             facecolor=color, 
#                             alpha=0.8,
#                             edgecolor='none')
#     ax.add_patch(rect)
    
#     # 添加文本
#     ax.text(start + width/2, 0.5, word, 
#             ha='center', va='center', 
#             fontsize=6, 
#             color='black' if norm_val < 0.7 else 'white')  # 深色背景用白色文本
    
#     # 可选：在矩形下方显示数值
#     # ax.text(start + width/2, 0.15, f"{val:.2f}", 
#     #         ha='center', va='center', 
#     #         fontsize=8, 
#     #         color='black' if norm_val < 0.7 else 'white')

# plt.tight_layout()
# plt.savefig('text_fixed.png', dpi=300, bbox_inches='tight')
# # plt.show()





#####  给出一句话， 提取里面每一个token上对应的前五个特征的index 以及 激活值

# dataset_path = "MLLMMU/MLLMU-Bench"
# forget_dataset = load_dataset(dataset_path, "Full_Set")['train']
# # retain_dataset = load_dataset(dataset_path, "retain_90")['train']

# with open("dataset/famous_people.json", "w", encoding="utf-8") as f:
#     for forget_index in range(len(forget_dataset)):
#         # forget_index = 0
#         forget_image = forget_dataset[forget_index]['image']
#         forget_biography = forget_dataset[forget_index]['biography']
#         # forget_biography = json.loads(forget_dataset[forget_index]['biography'])
#         forget_conversation = conversation_form(forget_biography)
#         forget_prompt = model.processor.apply_chat_template(forget_conversation, add_generation_prompt=True)
#         forget_inputs = model.processor(images=forget_image, text=forget_prompt, return_tensors='pt').to(0, torch.float16)
#         forget_model_activations = get_model_activations(model, forget_inputs, sparse_autoencoder.cfg)

#         forget_token_ids = model.processor.tokenizer(forget_prompt)['input_ids']
#         forget_text_list = [model.processor.tokenizer.decode([tid], skip_special_tokens=False) for tid in forget_token_ids]

#         hook_name = "hook_hidden_post"
#         forget_sae_activations = sparse_autoencoder.run_with_cache(forget_model_activations)[1][hook_name]
#         feature_sae_activations = forget_sae_activations[0]
#         values, indices = topk(feature_sae_activations, k = 5, dim = 1)


#         res = {}
#         for index in range(len(forget_text_list)):
#             key = forget_text_list[index]
#             value = values[index, :].detach().cpu().numpy().tolist()
#             indice = indices[index, :].detach().cpu().numpy().tolist()
#             res[key] = {"sae_value": value, "sae_indice": indice}

#         json.dump(res, f, ensure_ascii=False, indent=2)

# feature_index = 39356
# feature_sae_activations = forget_sae_activations[0].transpose(0,1)[feature_index, :].detach().cpu().numpy()








# def load_sae_model(sae_path):
#     sae_path = sae_path
#     loaded_object = torch.load(sae_path)
#     cfg = loaded_object['cfg']
#     state_dict = loaded_object['state_dict']

#     sparse_autoencoder = SparseAutoencoder(cfg)
#     sparse_autoencoder.load_state_dict(state_dict)
#     sparse_autoencoder.eval()

#     loader = ViTSparseAutoencoderSessionloader(cfg)
#     model = loader.get_model(cfg.model_name)
#     model.to(cfg.device)
    
#     return sparse_autoencoder, model


# def generate_adj_sae_outputs(sparse_autoencoder, model, raw_image, text_prompt, label, max_token):
    
#     # res_file = f"{directory}/adj_sae_outputs.json"
    
#     # existing_labels = []
#     # if os.path.exists(res_file):
#     #     with open(res_file, "r", encoding="utf-8") as f:
#     #         for line in f:
#     #             line = json.loads(line)
#     #             existing_labels.append(line['label'])

#     #     if label in existing_labels:
#     #         print(f"Label '{label}' Already exists, skip adding.")
#     #         return
            
#     conversation = [
#         {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": text_prompt},   # "What are these?" "Please describe this picture."
#             {"type": "image"},
#             ],
#         },
#     ]
    
#     prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
#     model_inputs = model.processor(images=raw_image, text=prompt, return_tensors='pt').to(sparse_autoencoder.device, torch.float16)
    
#     input_ids = model_inputs.input_ids
#     attention_mask = model_inputs.attention_mask
#     pixel_values = model_inputs.pixel_values
#     # image_sizes = model_inputs.image_sizes

#     tokenizer = model.processor.tokenizer
#     prompt_str_tokens = tokenizer.convert_ids_to_tokens(input_ids[0]) 
#     # answer_str_tokens = tokenizer.convert_ids_to_tokens(answer_tokens[0])

#     max_token = max_token
#     generated_ids = input_ids.clone()


#     def sae_hook(activations):
#         activations[:,0:576,:] = sparse_autoencoder(activations[:,0:576,:], label)[0]    
#         return (activations,)

#     sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)]

#     new_tokens = []
#     with torch.no_grad():
#         for ele in range(max_token):
#             print(f"Token: {ele}")
#             outputs = model.run_with_hooks(
#                 sae_hooks,
#                 return_type='output',
#                 input_ids=generated_ids,
#                 attention_mask=attention_mask,
#                 pixel_values=pixel_values,
#                 # image_sizes=image_sizes,
#             )
            
#             logits = outputs.logits[:, -1, :]  
#             next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
#             generated_ids = torch.cat([generated_ids, next_token], dim=-1)
#             new_tokens.append(next_token)
#             new_mask = torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
#             attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
#             torch.cuda.empty_cache()

#     output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#     print(output_texts)
    
#     # res = {"label": label, "instruction": "Please describe this picture.", "output_texts": output_texts}
#     # with open(res_file, "a", encoding="utf-8") as f:
#     #     f.write(json.dumps(res, ensure_ascii=False) + "\n") 


# def generate_original_outputs(model, processor, raw_image, text_prompt, max_token=306):
    
#     conversation = [
#         {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": text_prompt},  # "What are these?"  "Please describe the objects in the image and their corresponding colors."
#             {"type": "image"},
#             ],
#         },
#     ]

#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     model_inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
#     output = model.generate(**model_inputs, max_new_tokens=max_token, do_sample=False)

#     output_texts = processor.decode(output[0][2:], skip_special_tokens=True)
#     print(output_texts)
    

