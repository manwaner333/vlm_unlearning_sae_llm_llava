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

from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizerFast, AutoTokenizer, AutoProcessor, MllamaForConditionalGeneration, AutoModelForImageTextToText
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

def sae_hook(activations):
    # activations[:,-1,:] = sparse_autoencoder(activations[:,-1,:])[0]
    activations[:,-1,:] = activations[:,-1,:]
    return (activations,)

def generate_image_text(model, processor, conversation, image, max_token):
    sentence_end_pattern = re.compile(r"[.?!]\s*$")
    with torch.no_grad():
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        model_inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        pixel_values = model_inputs.pixel_values
        generated_ids = input_ids.clone()
                
        # sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)] 
        print("test case:")
        for ele in range(max_token):
            outputs = model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                # image_sizes=image_sizes,
            )
            logits = outputs.logits[:, -1, :]  
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            
            if next_token == model.config.eos_token_id:
                break
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            new_mask = torch.ones((attention_mask.shape[0], 1), device=model.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            
            decoded_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if sentence_end_pattern.search(decoded_text):
                break
    
            torch.cuda.empty_cache()

        output_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return output_texts

def generate_text(model, processor, conversation, max_token):
    sentence_end_pattern = re.compile(r"[.?!]\s*$")
    with torch.no_grad():
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        model_inputs = processor(text=prompt, return_tensors='pt').to(0, torch.float16)
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        # pixel_values = model_inputs.pixel_values
        generated_ids = input_ids.clone()
                
        # sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)] 
        print("test case:")
        for ele in range(max_token):
            outputs = model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                # pixel_values=pixel_values,
                # image_sizes=image_sizes,
            )
            logits = outputs.logits[:, -1, :]  
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            
            if next_token == model.config.eos_token_id:
                break
    
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            new_mask = torch.ones((attention_mask.shape[0], 1), device=model.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            
            decoded_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if sentence_end_pattern.search(decoded_text):
                break
            
            torch.cuda.empty_cache()

        output_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return output_texts
        
def evaluate_classification(classification_task, image, model, processor, id):
    print("################################## Classification Task Starts ##############################################")
    print("################################## Image Textual Questions ##############################################")
    image_textual_correct = 0
    image_textual_questions = 0
    pure_text_correct = 0
    pure_text_questions = 0
    max_token = 2
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
                      f"Just give ONE letter representing the answer directly.")
        conversation = conversation_form(combined_question)
        output_texts = generate_image_text(model, processor, conversation, image, max_token)
        
        if "ASSISTANT:" in output_texts:
            assistant_response = output_texts.split("ASSISTANT:")[1].strip()
        elif "Answer:" in output_texts:
            assistant_response = output_texts.split("Answer:")[1].strip()
        else:
            assistant_response = output_texts.strip()
        
        predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
        
        print("id: ", id)    
        print("Prompt: ", combined_question)
        print("Model Answer: ", predicted_answer)
        print("Correct Answer: ", correct_answer)
        print("The model answer is: ", predicted_answer == correct_answer)   # correct_answer.lower() in assistant_response.lower() or assistant_response.lower() in correct_answer.lower()
        print("\n")
        
        # if correct_answer.lower() in assistant_response.lower() or assistant_response.lower() in correct_answer.lower():
        #     image_textual_correct += 1
        # image_textual_questions += 1
        
        if predicted_answer == correct_answer:
            image_textual_correct += 1
        image_textual_questions += 1
        
        result = {
                "id": id,
                "question type": "Image_Textual",
                "question": question,
                "model_answer": predicted_answer,
                "ground_truth": correct_answer,
                "prompt": combined_question,
                "generated_answer": output_texts,
                "image_textual_correct": image_textual_correct,
                "image_textual_questions": image_textual_questions,
                "pure_text_correct": pure_text_correct,
                "pure_text_questions": pure_text_questions
            }
            
        with open(f'{output_folder}/{output_file}_classification_results_official.json', 'a') as f:
            f.write(json.dumps(result) + "\n")
        
        
    print("################################## Textual Questions ##############################################")
    for ele in Classification_Task['Pure_Text_Questions']:
        correct_answer = ele['Correct_Answer']
        options = ele['Options']
        question = ele['Question']
        
        question_with_options = formulate_prompt_with_options(question, options)
        combined_question = (f"{question_with_options}"
                      f"Just give ONE letter representing the answer directly.")
        conversation = conversation_only_text_form(combined_question)
        output_texts =  generate_text(model, processor, conversation, max_token)
        
        if "ASSISTANT:" in output_texts:
            assistant_response = output_texts.split("ASSISTANT:")[1].strip()
        elif "Answer:" in output_texts:
            assistant_response = output_texts.split("Answer:")[1].strip()
        else:
            assistant_response = output_texts.strip()
            
        predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
        
        print("id: ", id)    
        print("Prompt: ", combined_question)
        print("Model Answer: ", predicted_answer)
        print("Correct Answer: ", correct_answer)
        print("The model answer is: ", predicted_answer == correct_answer)
        print("\n")
        
        # if correct_answer.lower() in assistant_response.lower() or assistant_response.lower() in correct_answer.lower():
        #     pure_text_correct += 1
        # pure_text_questions += 1
        
        if predicted_answer == correct_answer:
            pure_text_correct += 1
        pure_text_questions += 1
        
        result = {
                "id": id,
                "question type": "Pure_Text",
                "question": question,
                "model_answer": assistant_response,
                "ground_truth": correct_answer,
                "prompt": combined_question,
                "generated_answer": output_texts,
                "image_textual_correct": image_textual_correct,
                "image_textual_questions": image_textual_questions,
                "pure_text_correct": pure_text_correct,
                "pure_text_questions": pure_text_questions
            }
            
        with open(f'{output_folder}/{output_file}_classification_results_official.json', 'a') as f:
            f.write(json.dumps(result) + "\n")
        
    return image_textual_correct, image_textual_questions, pure_text_correct, pure_text_questions

def compute_bleu(ground_truth, predicted_answer):
    """
    Compute the BLEU score between a ground truth and predicted answer using simple whitespace tokenization.

    Args:
        ground_truth (str): The correct reference answer.
        predicted_answer (str): The predicted answer from the model.

    Returns:
        float: The BLEU score.
    """
    # Use .split() to tokenize based on spaces
    reference = [ground_truth.split()]  # Reference needs to be a list of tokenized words
    hypothesis = predicted_answer.split()  # Hypothesis (predicted answer) is also tokenized

    # Use smoothing to handle cases where BLEU score could be 0 for short texts
    smoothing_function = SmoothingFunction().method1

    # Compute the BLEU score
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

    return bleu_score


def evaluate_generation(generation_task, image, model, processor, id):
    print("################################## Generation Task Starts ##############################################")
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_img = 0
    rouge1_img = 0
    rouge2_img = 0
    rougeL_img = 0
    image_textual_questions = 0
    bleu_text = 0
    rouge1_text = 0
    rouge2_text = 0
    rougeL_text = 0
    pure_text_questions = 0
    max_token = 20
    
    for ele in generation_task:
        ground_truth = ele['Ground_Truth']
        type = ele['Type']
        question = ele['Question']
        
        combined_question = f"""Answer the following question based on your trained knowledge in one sentence accurately in ENGLISH.
        Question: {question}
        Answer:
        """
        
        # combined_question = f"{question}\nAnswer the following question based on your trained knowledge in one sentence accurately in ENGLISH."
        
        if type == 'Image_Textual':
            conversation = conversation_form(combined_question)
            output_texts = generate_image_text(model, processor, conversation, image, max_token)
        else:
            conversation = conversation_only_text_form(combined_question)
            output_texts =  generate_text(model, processor, conversation, max_token)
        
        if "ASSISTANT:" in output_texts:
            predicted_answer = output_texts.split("ASSISTANT:")[1].strip()
        elif "Answer:" in output_texts:
            predicted_answer = output_texts.split("Answer:")[1].strip()
        else:
            predicted_answer = output_texts.strip()
            
        print("Prompt: ", combined_question)
        print("Question: ", question)
        print("question type: ",  type)
        print("Model Answer: ", predicted_answer)
        print("Correct Answer: ", ground_truth)
        print("\n")
        
        # doc = nlp(predicted_answer)
        # first_sentence = [sent.text for sent in doc.sents][0]

        bleu_score = compute_bleu(ground_truth, predicted_answer)
        rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

        if type == "Image_Textual":
            # Accumulate scores for Image_Textual questions
            bleu_img += bleu_score
            rouge1_img += rouge_scores['rouge1'].fmeasure
            rouge2_img += rouge_scores['rouge2'].fmeasure
            rougeL_img += rouge_scores['rougeL'].fmeasure
            image_textual_questions += 1
        else:
            # Accumulate scores for Pure_Text questions
            bleu_text += bleu_score
            rouge1_text += rouge_scores['rouge1'].fmeasure
            rouge2_text += rouge_scores['rouge2'].fmeasure
            rougeL_text += rouge_scores['rougeL'].fmeasure
            pure_text_questions += 1
                
        result = {
            "id": id,
            "question type": type,
            "question": question,
            "model_answer": predicted_answer,
            "ground_truth": ground_truth,
            "prompt": combined_question,
            "generated_answer": output_texts,
            "bleu_img": bleu_img,
            "rouge1_img": rouge1_img,
            "rouge2_img": rouge2_img,
            "rougeL_img": rougeL_img,
            "image_textual_questions": image_textual_questions,
            "bleu_text": bleu_text,
            "rouge1_text": rouge1_text,
            "rouge2_text": rouge2_text,
            "rougeL_text": rougeL_text,
            "pure_text_questions": pure_text_questions
            }
        
        with open(f'{output_folder}/{output_file}_generation_results_official.json', 'a') as f:
            f.write(json.dumps(result) + "\n")
    
    return   bleu_img, rouge1_img, rouge2_img, rougeL_img, image_textual_questions, bleu_text, rouge1_text, rouge2_text, rougeL_text, pure_text_questions
        
        
        
def evaluate_fill_blank(mask_task, image, model, processor, id):
    print("################################## Fill-in-the-blank Task Starts ##############################################")
    image_textual_correct = 0
    image_textual_questions = 0
    pure_text_correct = 0
    pure_text_questions = 0
    for ele in mask_task:
        question = ele["Question"]
        ground_truth = ele["Ground_Truth"]
        question_type = ele["Type"]
        
        combined_question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
        max_token = 20
        if question_type == "Image_Textual":
            conversation = conversation_form(combined_question)
            output_texts = generate_image_text(model, processor, conversation, image, max_token)
        else:
            conversation = conversation_only_text_form(combined_question)
            output_texts =  generate_text(model, processor, conversation, max_token)
            
        if "ASSISTANT:" in output_texts:
            assistant_response = output_texts.split("ASSISTANT:")[1].strip()
        elif "Answer:" in output_texts:
            assistant_response = output_texts.split("Answer:")[1].strip()
        else:
            assistant_response = output_texts.strip()
        
        print("id: ", id)    
        print("Prompt: ", combined_question)
        print("Model Answer: ", assistant_response)
        print("Correct Answer: ", ground_truth)
        print("The model answer is: ", ground_truth.lower() in assistant_response.lower()) # or assistant_response.lower() in ground_truth.lower()
        print("\n")
        
        if question_type == "Image_Textual":
            if ground_truth.lower() in assistant_response.lower(): # fill_blank_pure_text_questions_total
                image_textual_correct += 1
            image_textual_questions += 1
        elif question_type == "Pure_Text":
            if ground_truth.lower() in assistant_response.lower(): # or assistant_response.lower() in ground_truth.lower()
                pure_text_correct += 1
            pure_text_questions += 1
        
        result = {
                "id": id,
                "question type": question_type,
                "question": question,
                "model_answer": assistant_response,
                "ground_truth": ground_truth,
                "prompt": combined_question,
                "generated_answer": output_texts,
                "image_textual_correct": image_textual_correct,
                "image_textual_questions": image_textual_questions,
                "pure_text_correct": pure_text_correct,
                "pure_text_questions": pure_text_questions
            }
            
        with open(f'{output_folder}/{output_file}_fill_blank_results_official.json', 'a') as f:
            f.write(json.dumps(result) + "\n")
       
    
    return image_textual_correct,  image_textual_questions, pure_text_correct, pure_text_questions    
        
 
 

### load sparse autoencoder
# sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/c56dd1601694cfb7a43202199b0f25a4b617a83b/32954364_pre_trained_llava_sae_language_model_65536.pt"
# sparse_autoencoder, model = load_sae_model(sae_path)
# sparse_autoencoder.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("jiahuimbzuai/llava_vanilla_model")
model = AutoModelForImageTextToText.from_pretrained("jiahuimbzuai/llava_vanilla_model", torch_dtype=torch.bfloat16, device_map="auto")
model.to(device)
model.eval()
 
### load dataset
dataset_path = "MLLMMU/MLLMU-Bench"
forget_dataset = load_dataset(dataset_path, "forget_10")['train']
retain_dataset = load_dataset(dataset_path, "retain_90")['train']

  

fill_blank_image_textual_correct_total = 0
fill_blank_image_textual_questions_total = 0
fill_blank_pure_text_correct_total = 0
fill_blank_pure_text_questions_total = 0 
output_folder = 'result/llava_1.5_7b_vanilla_model_forget_10'
output_file = 'llava_1.5_7b_vanilla_model_forget_10'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
classification_image_textual_correct_total = 0
classification_image_textual_questions_total = 0
classification_pure_text_correct_total = 0
classification_pure_text_questions_total = 0 

generation_bleu_img_total = 0
generation_rouge1_img_total = 0
generation_rouge2_img_total = 0
generation_rougeL_img_total = 0
generation_image_textual_questions_total = 0
generation_bleu_text_total = 0
generation_rouge1_text_total = 0
generation_rouge2_text_total = 0
generation_rougeL_text_total = 0
generation_pure_text_questions_total = 0   

   
for index in range(len(forget_dataset)):  # [0,1,2]:  # range(0, 50):
    id = forget_dataset[index]['ID']
    print(f"ID: {id}")
    image = forget_dataset[index]['image']
    biography = forget_dataset[index]['biography']
    question = forget_dataset[index]['question']
    answer = forget_dataset[index]['answer']
    Classification_Task = forget_dataset[index]['Classification_Task']
    Generation_Task = forget_dataset[index]['Generation_Task']
    Mask_Task = forget_dataset[index]['Mask_Task']
    
    ### classification_task
    # classification_image_textual_correct, classification_image_textual_questions, classification_pure_text_correct, classification_pure_text_questions = evaluate_classification(Classification_Task, image, model, processor, id)
    # classification_image_textual_correct_total += classification_image_textual_correct
    # classification_image_textual_questions_total += classification_image_textual_questions
    # classification_pure_text_correct_total += classification_pure_text_correct
    # classification_pure_text_questions_total += classification_pure_text_questions 
    
    ### generation task
    # bleu_img, rouge1_img, rouge2_img, rougeL_img, image_textual_questions, bleu_text, rouge1_text, rouge2_text, rougeL_text, pure_text_questions = evaluate_generation(Generation_Task, image, model, processor, id)
    # generation_bleu_img_total += bleu_img
    # generation_rouge1_img_total += rouge1_img
    # generation_rouge2_img_total += rouge2_img
    # generation_rougeL_img_total += rougeL_img
    # generation_image_textual_questions_total += image_textual_questions
    # generation_bleu_text_total += bleu_text
    # generation_rouge1_text_total += rouge1_text
    # generation_rouge2_text_total += rouge2_text
    # generation_rougeL_text_total += rougeL_text
    # generation_pure_text_questions_total += pure_text_questions
    
    # ### fill_blank_task 
    image_textual_correct,  image_textual_questions, pure_text_correct, pure_text_questions = evaluate_fill_blank(Mask_Task, image, model, processor, id)
    fill_blank_image_textual_correct_total += image_textual_correct
    fill_blank_image_textual_questions_total += image_textual_questions
    fill_blank_pure_text_correct_total += pure_text_correct
    fill_blank_pure_text_questions_total += pure_text_questions
        

### generation task
# avg_scores = {}
# if generation_image_textual_questions_total > 0:
#     avg_scores.update({
#         "Average ROUGE-1 (Image_Textual)": generation_rouge1_img_total / generation_image_textual_questions_total,
#         "Average ROUGE-2 (Image_Textual)": generation_rouge2_img_total / generation_image_textual_questions_total,
#         "Average ROUGE-L (Image_Textual)": generation_rougeL_img_total / generation_image_textual_questions_total,
#         "Average BLEU (Image_Textual)": generation_bleu_img_total / generation_image_textual_questions_total
#     })

# if generation_pure_text_questions_total > 0:
#     avg_scores.update({
#         "Average ROUGE-1 (Pure_Text)": generation_rouge1_text_total / generation_pure_text_questions_total,
#         "Average ROUGE-2 (Pure_Text)": generation_rouge2_text_total / generation_pure_text_questions_total,
#         "Average ROUGE-L (Pure_Text)": generation_rougeL_text_total / generation_pure_text_questions_total,
#         "Average BLEU (Pure_Text)": generation_bleu_text_total / generation_pure_text_questions_total
#     })

# for metric, score in avg_scores.items():
#     print(f"{metric}: {score}")  



### classification task
# classification_image_textual_accuracy = (classification_image_textual_correct_total / classification_image_textual_questions_total) * 100 if classification_image_textual_questions_total > 0 else 0
# classification_pure_text_accuracy = (classification_pure_text_correct_total / classification_pure_text_questions_total) * 100 if classification_pure_text_questions_total > 0 else 0


# print(f"classification_image_textual_correct_total: {classification_image_textual_correct_total}")
# print(f"classification_image_textual_questions_total: {classification_image_textual_questions_total}")
# print(f"classification_pure_text_correct_total: {classification_pure_text_correct_total}")
# print(f"classification_pure_text_questions_total: {classification_pure_text_questions_total}")
    
# print(f"Classification Image-Textual Question Accuracy: {classification_image_textual_accuracy:.2f}%")
# print(f"Classification Pure Text Question Accuracy: {classification_pure_text_accuracy:.2f}%")


### fill blank task
fill_blank_image_textual_accuracy = (fill_blank_image_textual_correct_total / fill_blank_image_textual_questions_total) * 100 if fill_blank_image_textual_questions_total > 0 else 0
fill_blank_pure_text_accuracy = (fill_blank_pure_text_correct_total / fill_blank_pure_text_questions_total) * 100 if fill_blank_pure_text_questions_total > 0 else 0

print(f"total_image_textual_correct: {fill_blank_image_textual_correct_total}")
print(f"total_image_textual_questions: {fill_blank_image_textual_questions_total}")
print(f"total_pure_text_correct: {fill_blank_pure_text_correct_total}")
print(f"total_pure_text_questions: {fill_blank_pure_text_questions_total}")
print(f"fill blank Image-Textual Question Accuracy: {fill_blank_image_textual_accuracy:.2f}%")
print(f"fill blank Pure Text Question Accuracy: {fill_blank_pure_text_accuracy:.2f}%")