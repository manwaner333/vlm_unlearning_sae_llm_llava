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
from vit_sae_analysis.dashboard_fns import get_feature_data


sys.path.append("..")

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import ViTSparseAutoencoderSessionloader

if torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
# sae_path = "checkpoints/0ns2guf8/final_sparse_autoencoder_llava-hf/llava-1.5-7b-hf_-2_resid_131072.pt"
sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/424fb7f12fba943f7b029262f6fb1d9c2f0f3262/131815620_pre_trained_llava_sae_language_model_65536_update.pt"

loaded_object = torch.load(sae_path)

cfg = loaded_object['cfg']

state_dict = loaded_object['state_dict']

sparse_autoencoder = SparseAutoencoder(cfg)

sparse_autoencoder.load_state_dict(state_dict)

sparse_autoencoder.eval()

loader = ViTSparseAutoencoderSessionloader(cfg)

model = loader.get_model(cfg.model_name)

model.to(cfg.device)

get_feature_data(
    sparse_autoencoder,
    model,
    number_of_images = 1000,  # 524_288,
    number_of_max_activating_images = 10,
    max_number_of_images_per_iteration = 20,
)