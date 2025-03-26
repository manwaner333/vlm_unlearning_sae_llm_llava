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
from transformers import AutoProcessor, MllamaForConditionalGeneration, AutoModelForImageTextToText
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

model_path = "jiahuimbzuai/llava_vanilla_model"
# model_path = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    
dataset_path = 'MLLMMU/MLLMU-Bench'
dataset = load_dataset(dataset_path, "forget_10")['train']    # Retain_Set Full_Set

# print(f"Total data quantity: {len(dataset)}")
# for index in range(len(dataset)):
    
#     image = dataset[index]['image']

#     biography = dataset[index]['biography']
#     question = dataset[index]['question']
#     answer = dataset[index]['answer']
#     Classification_Task = dataset[index]['Classification_Task']
#     Generation_Task = dataset[index]['Generation_Task']
#     Mask_Task = dataset[index]['Mask_Task'] 

#     conversation = conversation_form(question)
#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

#     output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

#     print("Generated Answer:")
#     print(processor.decode(output[0][2:], skip_special_tokens=True))
#     print("Original Answer:")
#     print(answer)
#     print("----------------------------------------------------")

index = 0
image = dataset[index]['image']
biography = dataset[index]['biography']
question = dataset[index]['question']
answer = dataset[index]['answer']
Classification_Task = dataset[index]['Classification_Task']

for ele in Classification_Task['Image_Textual_Questions']:
    correct_answer = ele['Correct_Answer']
    options = ele['Options']
    question = ele['Question']
    combined_question = f"""You will be given a Question and multiple Options. Please choose the correct answer from the given Options.
    
    The answer must be one of the following: A, B, C, or D. Do not provide explanations, just output a single letter.
    
    Respond only with one letter: A, B, C, or D.

    Question: {question}
    Options: {options}
    Answer:
    """
    
    conversation = conversation_form(combined_question)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    model_inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
    output = model.generate(**model_inputs, max_new_tokens=2, do_sample=False)
    
    print("Generated Answer:")
    print(processor.decode(output[0][2:], skip_special_tokens=True))
    print("Original Answer:")
    print(correct_answer)
    print("----------------------------------------------------")