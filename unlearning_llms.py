# llm 的遗忘
import torch as t, einops
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache, FactoredMatrix
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
from IPython.display import display, HTML
#import circuitsvis as cv
#from circuitsvis.attention import attention_heads
import numpy as np
from tqdm.notebook import tqdm
from tqdm import trange
from dataclasses import dataclass
from fancy_einsum import einsum
from functools import partial
from typing import List, Optional, Union
import plotly.express as px
import plotly.io as pio
from IPython.display import HTML, IFrame
from jaxtyping import Float

from transformer_lens.utils import (
    load_dataset,
    tokenize_and_concatenate,
    download_file_from_hf,
)

device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Device: {device}")
t.set_grad_enabled(False)
print("Disabled automatic differentiation")

# load model
model = HookedTransformer.from_pretrained(
    "gpt2",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
)

# Load Auto Encoder
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = 'gpt2-small-res-jb',#"gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.9.hook_resid_pre", # won't always be a hook point
    device = device
)

hp_prompts = [
    ("Harry Potter studies", "magic"),
    ("Harry Potter studies at","Hogwarts"),
    ("Harry Potter studies at the", "school"),
    ("Harry Potter's biggest enemy is","the Dark Lord"),
    ("Harry Potter's best friends are Ron and","Hermione"),
    ("The Harry Potter series was written by","J.K. Rowling"),
    ("J.K. Rowling is the author of the book series \"", "Harry Potter"),
    ("The boy who lived is a wizard named","Harry Potter") ,
    ("The Sorcerer's Stone is the first book in a series about a boy named","Harry Potter"),
    ("Hurrying along Diagon Alley, they stopped before the imposing building run by goblins, which every wizard knew as", "Gringotts"),
    ("As Harry Potter went up the headmaster's tower, looking forward to finally tell Professor","Dumbledore"),
    ("In the Defense against the Dark Arts class, he felt the scar on his","forehead"),
    ("The Quidditch match was intense, with both Seekers racing to catch the","Snitch"),
    ("Harry walked up to Professor Severus", "Snape"),
    ("Harry Potter is a boy and a", "wizard"),
    ("Names of famous fictional schools: 1.", "Hogwarts"),
    ("If someone wants to take classes on the Dark Arts, they should look for Professor", "Snape"),
    ("Stepping cautiously into the forbidden corridor, they heard the snores of the three-headed dog whose name they later discovered was \"","Fluffy"),
   ("After the loud pops of apparating Death Eaters, a chilling symbol appeared in the night sky, known among the wizards as the","Dark Mark"),
    ("Digging through the potions book, they found a complicated recipe that, when brewed correctly, allows one to assume the appearance of someone else, called the","Polyjuice Potion"),
    ("Discussing the specifics of their wands, they mentioned how one of the rarest cores a wand could possess was from a","phoenix feather"),
    ("When they needed it most, a hidden room appeared within the castle, its interior changing based on their requirements. This magical place was known as the","room of requirement"),
    ("During the intense Quidditch match, the seeker's eyes were locked on the elusive golden snitch as it darted around, waiting for the perfect moment to seize the golden", "Snitch"),
    ("In order to blend in with the non-magical population, they had to dress as","muggles"),
    ("One of Harry Potter's rivals at school, who belonged to a different house, was named","Draco Malfoy"),
    ("The popular wizarding sport, played on broomsticks and involving various flying balls, is called","Quidditch"),
    ("As Ron and Hermione repotted the unusual plant, they had to wear earmuffs to protect themselves from its piercing scream, a sound produced by the", "mandrake"),
    ("Some of the most popular novels about wizards include: 1. \"","Harry Potter"),
    ("They were preparing to visit a special hospital for magical maladies and injuries, known as St.","Mungo's"),
    ("wise and beloved headmaster of the magical school, who guided Harry throughout his journey, was named", "Dumbledore"),
    ("When Harry went back to class, he saw that his best friends,","Ron and Hermione"),
]

prompt,answer = hp_prompts[0]
utils.test_prompt(prompt, answer, model)


def logits_to_next_str(logits, model = model):
  assert logits.shape[0] == 1
  logits = logits[:,-1,:].squeeze()
  next_tok = logits.argmax(dim = -1).item() # scalar
  return model.to_string(next_tok)

answers = {}
max_tokens = 10
for prompt, answer in hp_prompts:
  out = prompt
  completion = ""
  gen = 0

  while gen < max_tokens:
    tokens = model.to_tokens(out)
    logits = model(tokens)
    next_str = logits_to_next_str(logits)
    completion += next_str
    out += next_str
    gen+=1

  answers[prompt] = answer, completion

correct = 0

for key,val in answers.items():
  targ,complete = val
  if targ in complete:
    correct+=1
  print(val)

total = len(answers)
acc = correct/total
print(f"{acc:.2F}")


# Specific SAE Feature Ablation Can Degrade Previously Correct HP Answers
from functools import partial

def ablate_features(resid_post: Float[t.Tensor, "batch seq d_model"],
                 hook:HookPoint,feature_indxs, clamp):
  sae.eval()
  with t.no_grad():
    feature_acts = sae.encode(resid_post)
    # what if we ablated all relevant features? would need multiple layers
    for indx in feature_indxs:
      feature_acts[...,indx] = clamp
    sae_out = sae.decode(feature_acts)
    resid_post[:] = sae_out

  return sae_out

def unlearn(tokens, model = model, clamp = 0): #try clamping at different levels
    model.reset_hooks()
    hook_name = sae.cfg.hook_name
    logits = model.run_with_hooks(tokens,return_type = "logits", fwd_hooks =[
        (hook_name,
        partial(ablate_features,feature_indxs = [10130,776],clamp=clamp))])
    return logits
  
answers2 = {}

for prompt, answer in hp_prompts:
  out = prompt
  completion = ""
  gen = 0

  while gen < max_tokens:
    tokens = model.to_tokens(out)
    logits = unlearn(tokens,clamp = 6)
    next_str = logits_to_next_str(logits)
    completion += next_str
    out += next_str
    gen+=1

  answers2[prompt] = answer, completion

correct = 0

for key,val in answers2.items():
  targ,complete = val
  if targ in complete:
    correct+=1

total = len(answers2)
acc = correct/total
print("ACCURACY")
print(f"{acc:.2F}")

normal_answers = [val[1] for val in answers.values()]
unlearn_answers = [val[1] for val in answers2.values()]
prompts = [val[0] for val in hp_prompts]
comp = list(zip(prompts,normal_answers,unlearn_answers))
for prompt, pre,new in comp:
  print(prompt)
  print("Old Answer:", pre)
  print("New Answer:", new)
  print("---"*50)