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

sys.path.append("..")

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

sys.path.append("..")

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

sys.path.append("..")

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

from typing import Union, List, Optional
import torch

if torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

original_model = False
sae_model = True



### specific case study---sparse autoencoder model
if sae_model:
    cfg = ViTSAERunnerConfig(
        
        # Data Generating Function (Model + Training Distibuion)
        class_token = True,
        image_width = 224,
        image_height = 224,
        model_name = "llava-hf/llava-1.5-7b-hf", 
        module_name = "resid",
        block_layer = -2,
        dataset_path = "evanarlian/imagenet_1k_resized_256", 
        use_cached_activations = False,
        cached_activations_path = None,
        d_in = 4096,  # 1024,
        
        # SAE Parameters
        expansion_factor = 32,  # 64,
        b_dec_init_method = "mean",
        
        # Training Parameters
        lr = 0.0004,
        l1_coefficient = 0.00008,
        lr_scheduler_name="constantwithwarmup",
        batch_size = 2, # 1024,
        lr_warm_up_steps=500,
        total_training_tokens = 400,  # 20000, # 2_621_440,
        n_batches_in_store = 10, # 15,
        
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
        
        # Activation Store Parameters # 自己添加的
        max_batch_size_for_vit_forward_pass = 5,
        
        from_pretrained_path = 'checkpoints/0ns2guf8/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt'
    )


    # load model, sparse_autoencoder, activations_loader
    torch.cuda.empty_cache()
    if cfg.from_pretrained_path is not None:
        model, sparse_autoencoder, activations_loader = ViTSparseAutoencoderSessionloader.load_session_from_pretrained(cfg.from_pretrained_path)
        cfg = sparse_autoencoder.cfg
    else:
        loader = ViTSparseAutoencoderSessionloader(cfg)
        model, sparse_autoencoder, activations_loader = loader.load_session()
    
    print("model, sparse_auroencode and activations loading finish!!!")


    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
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
    model_inputs = model.processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
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

        
    max_token = 50
    generated_ids = input_ids.clone()
    
    def sae_hook(activations):
        activations[:,-1,:] = sparse_autoencoder(activations[:,-1,:], ghost_grad_neuron_mask)[0]   # activations[:,-1,:]*0.5     #   activations[:,0,:] = sparse_autoencoder(activations[:,0,:])[0]
        return (activations,)
    
    n_forward_passes_since_fired = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
    ghost_grad_neuron_mask = (n_forward_passes_since_fired > sparse_autoencoder.cfg.dead_feature_window).bool()
    sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)] 

    with torch.no_grad():
        for _ in range(max_token):
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
            new_mask = torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            torch.cuda.empty_cache()

    output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(output_texts)




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




### neuron_alighment
# from plotly import express as xp
# import torch
# import plotly.io as pio

# sparsity_tensor = torch.load('dashboard/sae_sparsity.pt').to('cpu')
# print((sparsity_tensor>10/131000).sum())
# sparsity_tensor = torch.log10(sparsity_tensor)
# fig = xp.histogram(sparsity_tensor)
# fig.write_image("histogram.png")



### scatter_plots
# def plot_alignment_fig(cos_sims):
#     example_fig = px.line(cos_sims.to('cpu'))
#     example_fig.show()

# example_neurons = [25081,25097,38764,10186,14061,22552,41781,774,886,2681]
# sae_path = 'checkpoints/0ns2guf8/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt'
# loaded_object = torch.load(sae_path)
# cfg = loaded_object['cfg']
# state_dict = loaded_object['state_dict']
# sparse_autoencoder = SparseAutoencoder(cfg)
# sparse_autoencoder.load_state_dict(state_dict)
# sparse_autoencoder.eval()
# loader = ViTSparseAutoencoderSessionloader(cfg)
# model = loader.get_model(cfg.model_name)
# model.to(cfg.device)


# mlp_out_weights = model.model.language_model.model.layers[cfg.block_layer].mlp.down_proj.weight.detach().transpose(0,1) # size [hidden_mlp_dimemsion, resid_dimension]
# penultimate_mlp_out_weights = model.model.language_model.model.layers[cfg.block_layer-2].mlp.down_proj.weight.detach().transpose(0,1)
# sae_weights = sparse_autoencoder.W_enc.detach() # size [resid_dimension, sae_dimension]
# sae_weights /= torch.norm(sae_weights, dim = 0, keepdim = True)
# mlp_out_weights /= torch.norm(mlp_out_weights, dim = 1, keepdim = True)
# penultimate_mlp_out_weights /= torch.norm(penultimate_mlp_out_weights, dim = 1, keepdim = True)
# sae_weights = sae_weights.to(torch.float16)
# cosine_similarities = mlp_out_weights @ sae_weights # size [hidden_mlp_dimemsion, sae_dimension]


# cosine_similarities =torch.abs(cosine_similarities)
# max_cosine_similarities = torch.max(cosine_similarities, 0).values.to('cpu') # size [sae_dimension]
# subset_max_cosine_similarities = max_cosine_similarities[example_neurons]
# print(subset_max_cosine_similarities)
# mean_max_cos_sim = max_cosine_similarities.mean()
# var_max_cos_sim = max_cosine_similarities.var()

# threshold = 0.18
# num_above_threshold = (max_cosine_similarities>threshold).sum()

# fig = px.histogram(max_cosine_similarities, title = "Histogram of max cosine similarities of SAE features with MLP out tensor.")
# # fig.update_xaxes(range=[0.07, 1])
# # fig.show()
# fig.write_image("aaa.png")


# random_weights = torch.randn(sae_weights.size(), device = sae_weights.device)
# random_weights /= torch.norm(random_weights, dim = 0, keepdim = True)
# random_weights = random_weights.to(torch.float16)
# cosine_similarities = torch.abs(mlp_out_weights @ random_weights) # size [hidden_mlp_dimemsion, sae_dimension]
# max_cosine_similarities = torch.max(cosine_similarities, 0).values.to('cpu') # size [sae_dimension]
# rand_mean_max_cos_sim = max_cosine_similarities.mean()
# rand_var_max_cos_sim = max_cosine_similarities.var()

# rand_fig = px.histogram(max_cosine_similarities, title = "Histogram of max cosine similarities of random vectors with MLP out tensor.")
# # rand_fig.update_xaxes(range=[0.07, 1])
# # rand_fig.show()
# fig.write_image("bbb.png")

# cosine_similarities = torch.abs(penultimate_mlp_out_weights @ random_weights) # size [hidden_mlp_dimemsion, sae_dimension]
# max_cosine_similarities = torch.max(cosine_similarities, 0).values.to('cpu') # size [sae_dimension]
# rand_mean_max_cos_sim = max_cosine_similarities.mean()
# rand_var_max_cos_sim = max_cosine_similarities.var()

# rand_fig = px.histogram(max_cosine_similarities, title = "Histogram of max cosine similarities of SAE out tensor with MLP out tensor in the layer before.")
# # rand_fig.update_xaxes(range=[0.07, 1])
# # rand_fig.show()
# fig.write_image("ccc.png")









