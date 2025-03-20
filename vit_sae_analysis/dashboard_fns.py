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
import shutil
from datasets import Dataset, Features, Value
from datasets import Image as dataset_Image

def conversation_form(key):
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": key},
            {"type": "image"},
            ],
        },
    ]
    return conversation

def load_images_and_convert_to_tensors(directory_path, device='cuda'):
    images_tensors = []
    activations = []
    supported_formats = '.png'  # Targeting PNG files
    
    for entry in os.listdir(directory_path):
        if entry.lower().endswith(supported_formats):
            img_path = os.path.join(directory_path, entry)
            img = Image.open(img_path)  # Ensure image is in RGB
            img_tensor = convert_images_to_tensor([img], device=device)
            images_tensors.append(img_tensor)
            activation = float(entry.split('_')[1].replace('.png',''))
            activations.append(torch.tensor([activation]))
    images_tensors = torch.concat(images_tensors, dim =0).to(device)
    activations = torch.concat(activations, dim =0).to(device)
    return images_tensors, activations

def convert_images_to_tensor(images, device='cuda'):
    """
    Convert a list of PIL images to a PyTorch tensor in RGB format with shape [B, C, H, W].

    Parameters:
    - images: List of PIL.Image objects.
    - device: The device to store the tensor on ('cpu' or 'cuda').

    Returns:
    - A PyTorch tensor with shape [B, C, H, W].
    """
    # Define a transform to convert PIL images (in RGB) to tensors
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # Convert image to RGB
        transforms.Resize((256, 256)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a torch tensor
    ])

    # Ensure each image is in RGB format, apply the transform, and move to the specified device
    tensor_list = [transform(img).to(device) for img in images]
    tensor_output = torch.stack(tensor_list, dim=0)

    return tensor_output

def delete_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


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

def get_all_model_activations(model, images, conversations, cfg):
    
    batch_of_prompts = []
    for ele in conversations:
        batch_of_prompts.append(model.processor.apply_chat_template(ele, add_generation_prompt=True))
    
    inputs = model.processor(images=images, text=batch_of_prompts, padding=True, return_tensors="pt").to(cfg.device)
    sae_batches = get_model_activations(model, inputs, cfg)   
    
    # sae_batches1 = torch.cat(sae_batches, dim = 0)
    sae_batches = sae_batches.reshape(-1, cfg.d_in)
    sae_batches = sae_batches.to(cfg.device)
    return sae_batches

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

def print_memory():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the current device
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        remaining_memory = total_memory-allocated_memory

        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {total_memory / (1024**3):.2f} GB")
        print(f"Allocated Memory: {allocated_memory / (1024**3):.2f} GB")
        print(f"Remaining Memory: {remaining_memory / (1024**3):.2f} GB")
    else:
        print("CUDA is not available.")
        
def save_highest_activating_images(max_activating_image_indices, max_activating_image_values, directory, dataset, image_key):
    assert max_activating_image_values.size() == max_activating_image_indices.size(), "size of max activating image indices doesn't match the size of max activing values."
    number_of_neurons, number_of_max_activating_examples = max_activating_image_values.size()
    for neuron in trange(number_of_neurons):
        neuron_dead = True
        for max_activating_image in range(number_of_max_activating_examples):
            if max_activating_image_values[neuron, max_activating_image].item()>0:
                if neuron_dead:
                    if not os.path.exists(f"{directory}/{neuron}"):
                        os.makedirs(f"{directory}/{neuron}")
                    neuron_dead = False
                image = dataset[int(max_activating_image_indices[neuron, max_activating_image].item())][image_key]
                image.save(f"{directory}/{neuron}/{max_activating_image}_{int(max_activating_image_indices[neuron, max_activating_image].item())}_{max_activating_image_values[neuron, max_activating_image].item():.4g}.png", "PNG")

def get_new_top_k(first_values, first_indices, second_values, second_indices, k):
    total_values = torch.cat([first_values, second_values], dim = 1)
    total_indices = torch.cat([first_indices, second_indices], dim = 1)
    new_values, indices_of_indices = topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
    return new_values, new_indices

@torch.inference_mode()
def get_feature_data(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedVisionTransformer,
    number_of_images: int = 32_768,
    number_of_max_activating_images: int = 10,
    max_number_of_images_per_iteration: int = 16_384,
    seed: int = 1,
    load_pretrained = False,
):
    '''
    Gets data that will be used to create the sequences in the HTML visualisation.

    Args:
        feature_idx: int
            The identity of the feature we're looking at (i.e. we slice the weights of the encoder). A list of
            features is accepted (the result will be a list of FeatureData objects).
        max_batch_size: Optional[int]
            Optionally used to chunk the tokens, if it's a large batch

        left_hand_k: int
            The number of items in the left-hand tables (by default they're all 3).
        buffer: Tuple[int, int]
            The number of tokens on either side of the feature, for the right-hand visualisation.

    Returns object of class FeatureData (see that class's docstring for more info).
    '''
    torch.cuda.empty_cache()
    sparse_autoencoder.eval()
    
    dataset = load_dataset(sparse_autoencoder.cfg.dataset_path, split="train")
    
    dataset = dataset.select(range(1000))
    
    image_key = 'image'
    image_label = 'conversations' 
    dataset = dataset.shuffle(seed = seed)
    directory = "dashboard"
    
    all_tokens = []
    output_file = "dataset_output_tokens.json"
    
    if load_pretrained:
        max_activating_image_indices = torch.load(f'{directory}/max_activating_image_indices.pt')
        max_activating_image_values = torch.load(f'{directory}/max_activating_image_values.pt')
    else:
        max_activating_image_indices = torch.zeros([sparse_autoencoder.cfg.d_sae, number_of_max_activating_images]).to(sparse_autoencoder.cfg.device)
        max_activating_image_values = torch.zeros([sparse_autoencoder.cfg.d_sae, number_of_max_activating_images]).to(sparse_autoencoder.cfg.device)
        sae_sparsity = torch.zeros([sparse_autoencoder.cfg.d_sae]).to(sparse_autoencoder.cfg.device)
        sae_mean_acts = torch.zeros([sparse_autoencoder.cfg.d_sae]).to(sparse_autoencoder.cfg.device)
        number_of_images_processed = 0
        while number_of_images_processed < number_of_images:
            torch.cuda.empty_cache()
            batch_of_images = []
            batch_of_conversations = []
            for i in range(max_number_of_images_per_iteration):
                # images = dataset[number_of_images_processed:number_of_images_processed + max_number_of_images_per_iteration][image_key]
                # labels = dataset[number_of_images_processed:number_of_images_processed + max_number_of_images_per_iteration][image_label]
                image = dataset[number_of_images_processed+i][image_key]
                label = dataset[number_of_images_processed+i][image_label]
                label_origin = " ".join(v["value"].replace("<image>\n", "") for v in label)
                label = label_origin
                tokens = model.processor.tokenizer(label)
                len_tokens = len(tokens.input_ids)
                while len_tokens < sparse_autoencoder.cfg.context_size:
                    label = label + label_origin
                    tokens = model.processor.tokenizer(label)
                    len_tokens = len(tokens.input_ids)
                final_tokens = tokens.input_ids[0:sparse_autoencoder.cfg.context_size]
                all_tokens.append(final_tokens)
                final_text = model.processor.tokenizer.decode(final_tokens)
                batch_of_images.append(image)
                batch_of_conversations.append(conversation_form(final_text))
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(all_tokens, f)
                f.write("\n")  
               
            model_activations = get_all_model_activations(model, batch_of_images, batch_of_conversations, sparse_autoencoder.cfg) # tensor of size [batch, d_resid]
            sae_activations = get_sae_activations(model_activations, sparse_autoencoder).transpose(0,1) # tensor of size [feature_idx, batch]
            del model_activations
            sae_mean_acts += sae_activations.sum(dim = 1)
            sae_sparsity += (sae_activations>0).sum(dim = 1)
            
            # Convert the images list to a torch tensor
            values, indices = topk(sae_activations, k = number_of_max_activating_images, dim = 1)   # 1  # sizes [sae_idx, images] is the size of this matrix correct?
            # indices += number_of_images_processed
            number = int(sae_activations.shape[1]/max_number_of_images_per_iteration)
            indices += number_of_images_processed * number 
            
            max_activating_image_values, max_activating_image_indices = get_new_top_k(max_activating_image_values, max_activating_image_indices, values, indices, number_of_max_activating_images)
            
            """
            Need to implement calculations for covariance matrix but it will need an additional 16 GB of memory just to store it (32 if I am batching I think...). Could it be added and stored on the CPU? Probs not...
            """
            number_of_images_processed += max_number_of_images_per_iteration
        
        sae_mean_acts /= sae_sparsity
        sae_sparsity /= number_of_images_processed
        
        # Check if the directory exists
        if not os.path.exists(directory):
            # Create the directory if it does not exist
            os.makedirs(directory)
            
        # compute the label tensor
        max_activating_image_label_indices = torch.tensor([dataset[int(index)]['label'] for index in tqdm(max_activating_image_indices.flatten(), desc = "getting image labels")])
        # Reshape to original dimensions
        max_activating_image_label_indices = max_activating_image_label_indices.view(max_activating_image_indices.shape)
        torch.save(max_activating_image_indices, f'{directory}/max_activating_image_indices.pt')
        torch.save(max_activating_image_values, f'{directory}/max_activating_image_values.pt')
        torch.save(max_activating_image_label_indices, f'{directory}/max_activating_image_label_indices.pt')
        torch.save(sae_sparsity, f'{directory}/sae_sparsity.pt')
        torch.save(sae_mean_acts, f'{directory}/sae_mean_acts.pt')
        # Should also save label information tensor here!!!
        
    save_highest_activating_images(max_activating_image_indices[:10,:10], max_activating_image_values[:10,:10], directory, dataset, image_key)  # 1000
