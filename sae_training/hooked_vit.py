import torch
import torch.nn as nn
import timm
import math
from transformers import LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizerFast, AutoTokenizer, AutoProcessor, MllamaForConditionalGeneration, AutoModelForImageTextToText
from typing import Callable
from contextlib import contextmanager
from typing import List, Union, Dict, Tuple
from functools import partial
from torch import Tensor
from torch.nn import functional as F
from jaxtyping import Float
  


# The Hook class does not currently only supports hooking on the following locations:
# 1 - residual stream post transformer block.
# 2 - mlp activations.
# More hooks can be added at a later date, but only post-module.
class Hook():
  def __init__(self, block_layer: int, module_name: str, hook_fn: Callable, return_module_output = True):
    self.path_dict = {
        'resid': '',
    }
    assert module_name in self.path_dict.keys(), f'Module name \'{module_name}\' not recognised.'
    self.return_module_output = return_module_output
    self.function = self.get_full_hook_fn(hook_fn)
    self.attr_path = self.get_attr_path(block_layer, module_name)

  def get_full_hook_fn(self, hook_fn: Callable):

    def full_hook_fn(module, module_input, module_output):
      hook_fn_output = hook_fn(module_output[0])
      if self.return_module_output:
        return module_output
      else:
        return hook_fn_output # Inexplicably, the module output is not a tensor of activaitons but a tuple (tensor,)...??

    return full_hook_fn
  
  
  def get_attr_path(self, block_layer: int, module_name: str) -> str:
    # attr_path = f'vision_model.transformer.layers[{block_layer}]'   # "meta-llama/Llama-3.2-11B-Vision-Instruct
    # attr_path = f'vision_tower.vision_model.encoder.layers[{block_layer}]'  # llava-1.5-7b-hf
    attr_path = f'language_model.model.layers[{block_layer}]'
    attr_path += self.path_dict[module_name]
    return attr_path
  
  def get_module(self, model):
    return self.get_nested_attr(model, self.attr_path)

  def get_nested_attr(self, model, attr_path):
    """
    Gets a nested attribute from an object using a dot-separated path.
    """
    module = model
    attributes = attr_path.split(".")
    for attr in attributes:
        if '[' in attr:
            # Split at '[' and remove the trailing ']' from the index
            attr_name, index = attr[:-1].split('[')
            module = getattr(module, attr_name)[int(index)]
        else:
            module = getattr(module, attr)
    return module



class HookedVisionTransformer():
  def __init__(self, model_name: str, device = 'cuda'):
    model, processor = self.get_ViT(model_name)
    # model, processor, tokenizer = self.get_ViT(model_name)
    self.model = model.to(device)
    self.processor = processor
    # self.tokenizer = tokenizer

  def get_ViT1(self, model_name):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor
  
  def get_ViT(self, model_name):
    # processor = LlavaNextProcessor.from_pretrained("llava-v1.6-vicuna-7b-hf")
    # model = LlavaNextForConditionalGeneration.from_pretrained("llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # tokenizer = CLIPTokenizerFast.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", from_slow=True)
    # processor = CLIPProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", tokenizer=tokenizer)   # CLIPImageProcessor  "openai/clip-vit-large-patch14-336"
    # processor.size = {"height": 336, "width": 336}
    # processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    # processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = LlavaNextForConditionalGeneration.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
    # processor.image_processor.size = {"height": 336, "width": 336}
    
    # processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    # model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16)
    
    # processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    # model = MllamaForConditionalGeneration.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", torch_dtype=torch.bfloat16)
    
    # MLLMMU/LLaVA_Vanilla
    processor = AutoProcessor.from_pretrained("jiahuimbzuai/llava_vanilla_model")
    model = AutoModelForImageTextToText.from_pretrained("jiahuimbzuai/llava_vanilla_model", torch_dtype=torch.bfloat16)
    
    return model, processor

  def run_with_cache(self, list_of_hook_locations: List[Tuple[int,str]], *args, return_type = "output", **kwargs):
    cache_dict, list_of_hooks = self.get_caching_hooks(list_of_hook_locations)
    with self.hooks(list_of_hooks) as hooked_model:
      with torch.no_grad():
        output = hooked_model(*args, **kwargs)
        
    if return_type=="output":
      return output, cache_dict
    if return_type=="loss":
      return self.contrastive_loss(output.logits_per_image, output.logits_per_text), cache_dict
    else:
      raise Exception(f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'.")

  def get_caching_hooks(self, list_of_hook_locations: List[Tuple[int,str]]):
    """
    Note that the cache dictionary is index by the tuple (block_layer, module_name).
    """
    cache_dict = {}
    list_of_hooks=[]
    def save_activations(name, activations):
      cache_dict[name] = activations.detach()
    for (block_layer, module_name) in list_of_hook_locations:
      hook_fn = partial(save_activations, (block_layer, module_name))
      hook = Hook(block_layer, module_name, hook_fn)
      list_of_hooks.append(hook)
    return cache_dict, list_of_hooks

  @torch.no_grad
  def run_with_hooks(self, list_of_hooks: List[Hook], *args, return_type = "output", **kwargs):
    with self.hooks(list_of_hooks) as hooked_model:
      with torch.no_grad():
        output = hooked_model(*args, **kwargs)
    if return_type=="output":
      return output
    if return_type=="loss":
      return self.contrastive_loss(output.logits_per_image, output.logits_per_text)
    else:
      raise Exception(f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'.")
  
    
  def contrastive_loss(self, logits_per_image: Float[Tensor, "n_images n_prompts"], logits_per_text: Float[Tensor, "n_prompts n_images"]): # Assumes square matrices
    assert logits_per_image.size()[0]==logits_per_image.size()[1], "The number of prompts does not match the number of images."
    batch_size = logits_per_image.size()[0]
    labels = torch.arange(batch_size).long().to(logits_per_image.device)
    image_loss = F.cross_entropy(logits_per_image, labels)
    text_loss = F.cross_entropy(logits_per_text, labels)
    total_loss = (image_loss + text_loss) / 2
    return total_loss

  @contextmanager
  def hooks(self, hooks: List[Hook]):
    """

    This is a context manager for running a model with hooks. The funciton adds 
    forward hooks to the model, and then returns the hooked model to be run with 
    a foward pass. The funciton then cleans up by removing any hooks.

    Args:

      model VisionTransformer: The ViT that you want to run with the forward hook

      hooks List[Tuple[str, Callable]]: A list of forward hooks to add to the model. 
        Each hook is a tuple of the module name, and the hook funciton.

    """
    hook_handles = []
    try:
      for hook in hooks:
        # Create a full hook funciton, with all the argumnets needed to run nn.module.register_forward_hook().
        # The hook functions are added to the output of the module.
        module = hook.get_module(self.model)
        handle = module.register_forward_hook(hook.function)
        hook_handles.append(handle)
      yield self.model
    finally:
      for handle in hook_handles:
        handle.remove()
            
  def to(self, device):
    self.model = self.model.to(device)

  def __call__(self, *args, return_type = 'output', **kwargs):
    return self.forward(*args, return_type = return_type, **kwargs)

  def forward(self, *args, return_type = 'output', **kwargs):
    if return_type=='output':
      return self.model(*args, **kwargs)
    elif return_type == 'loss':
      output = self.model(*args, **kwargs)
      return self.contrastive_loss(output.logits_per_image, output.logits_per_text)
    else:
      raise Exception(f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'.")
  
  def eval(self):
    self.model.eval()
    
  def train(self):
    self.model.train()