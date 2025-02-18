import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.config import ViTSAERunnerConfig
from tqdm import tqdm, trange
from transformers import LlavaNextProcessor
import json
# from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader 
from datasets import Dataset, Features, Value  #  Image
from datasets import Image as dataset_Image    

class ViTActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs. 
    """
    def __init__(
        self, cfg: ViTSAERunnerConfig, model: HookedVisionTransformer, create_dataloader: bool = True, train=True,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = load_dataset(self.cfg.dataset_path, split="train")
        
        if self.cfg.dataset_path=="cifar100": # Need to put this in the cfg
            self.image_key = 'img'
            self.label_key = 'fine_label'
        else:
            self.image_key = 'image'
            self.label_key = 'label'
            
        self.labels = self.dataset.features[self.label_key].names
        self.dataset = self.dataset.shuffle(seed=42)
        print(f"Total data quantity: {len(self.dataset)}")
        self.iterable_dataset = iter(self.dataset)
        
        # 之前自己添加的数据提取
        # self.data_path = self.cfg.dataset_path
        # try:
        #     with open(self.data_path, "r") as f:
        #         data_json = json.load(f)
        # except:
        #     with open(self.data_path, "r") as f:
        #         data_json = [json.loads(line) for line in f.readlines()]
        # data_json = data_json[:400]
        
        # dataset_dict = {
        #     "image": [item["image_path"] for item in data_json],
        #     "label": [item["name"] for item in data_json]
        # }
        
        # features = Features({
        #     "image": dataset_Image(),  # 指定 image_path 为 Image 类型
        #     "label": Value("string")
        # })
        
        # hf_dataset = Dataset.from_dict(dataset_dict, features=features)
        # self.dataset = hf_dataset
        # self.image_key = 'image'
        # self.label_key = 'label'
        
        # self.dataset = self.dataset.shuffle(seed=42)    
        # self.iterable_dataset = iter(self.dataset)
        # 自己添加的数据提取结束
        
        # imag_transform = transforms.Compose([
        #     transforms.Resize((224, 224)), 
        #     transforms.ToTensor(),    
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #     ])
        
        # def load_image(example):
        #     image = Image.open(example["image_path"]).convert("RGB")
        #     example["image"] = imag_transform(image)
        #     return example
        
        # self.dataset = hf_dataset.map(load_image)    
        
        if self.cfg.use_cached_activations:
            """
            Need to implement this. loads stored activations from a file.
            """
            pass
        
        if create_dataloader:
            if self.cfg.class_token:
              print("Starting to create the data loader!!!")
              self.dataloader = self.get_data_loader()
              print("Data loader created!!!")
            else:
              """
              Need to implement a buffer for the image patch training.
              """
              pass
          
    def get_batch_of_images_and_labels(self):
        batch_size = self.cfg.max_batch_size_for_vit_forward_pass
        images = []
        labels = []
        conversations = []
        for _ in range(batch_size):
            try:
                data = next(self.iterable_dataset)
            except StopIteration:
                self.iterable_dataset = iter(self.dataset.shuffle())
                data = next(self.iterable_dataset)
            image = data[self.image_key]
            label_index = "Please describe the content of this image."
            images.append(image)
            labels.append(label_index)
            conversations.append(self.conversation_form(label_index))
            # labels.append(f"A photo of a {self.labels[label_index]}.")
        
        batch_of_prompts = []
        for ele in conversations:
            batch_of_prompts.append(self.model.processor.apply_chat_template(ele, add_generation_prompt=True))
            
        inputs = self.model.processor(images=images, text=batch_of_prompts, return_tensors="pt", padding = True).to(self.cfg.device)
        return inputs
        

    # def get_image_batches(self):
    #     """
    #     A batch of tokens from a dataset.
    #     """
    #     device = self.cfg.device
    #     batch_of_images = []
    #     with torch.no_grad():
    #         for _ in trange(self.cfg.store_size, desc = "Filling activation store with images"):
    #             try:
    #                 batch_of_images.append(next(self.iterable_dataset)[self.image_key])
    #             except StopIteration:
    #                 self.iterable_dataset = iter(self.dataset.shuffle())
    #                 batch_of_images.append(next(self.iterable_dataset)[self.image_key])
    #     return batch_of_images
    
    def conversation_form(self, key):
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
    
    def get_image_conversation_batches(self):
        """
        A batch of tokens from a dataset.
        """
        device = self.cfg.device
        batch_of_images = []
        batch_of_conversations = []
        with torch.no_grad():
            for _ in trange(self.cfg.store_size, desc = "Filling activation store with images"):
                try:
                    image = next(self.iterable_dataset)[self.image_key]
                    # label = next(self.iterable_dataset)[self.label_key]
                    label = "Please describe the content of this image."
                    batch_of_images.append(image)
                    batch_of_conversations.append(self.conversation_form(label))
                except StopIteration:
                    self.iterable_dataset = iter(self.dataset.shuffle())
                    image = next(self.iterable_dataset)[self.image_key]
                    label = next(self.iterable_dataset)[self.label_key]
                    batch_of_images.append(image)
                    batch_of_conversations.append(self.conversation_form(label))
        return batch_of_images, batch_of_conversations
    

    # def get_activations(self, image_batches):
        
    #     module_name = self.cfg.module_name
    #     block_layer = self.cfg.block_layer
    #     list_of_hook_locations = [(block_layer, module_name)]

    #     # inputs = self.model.processor(images=image_batches, text = "", return_tensors="pt", padding = True).to(self.cfg.device)
        
    #     conversation = [
    #         {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "What is shown in this image?"}, 
    #             {"type": "image"},
    #             ],
    #         },
    #     ]
        
    #     prompt = self.model.processor.apply_chat_template(conversation, add_generation_prompt=True)
    #     inputs = self.model.processor(images=image_batches, text=prompt, return_tensors="pt").to(self.cfg.device)

    #     activations = self.model.run_with_cache(
    #         list_of_hook_locations,
    #         **inputs,
    #     )[1][(block_layer, module_name)]
        
    #     if self.cfg.class_token:
    #       # Only keep the class token
    #       activations = activations[:,0,:] # See the forward(), foward_head() methods of the VisionTransformer class in timm. 
    #       # Eg "x = x[:, 0]  # class token" - the [:,0] indexes the batch dimension then the token dimension

    #     return activations
    
    def get_activations(self, image_batches, conversation_batches):
        
        module_name = self.cfg.module_name
        block_layer = self.cfg.block_layer
        list_of_hook_locations = [(block_layer, module_name)]

        batch_of_prompts = []
        for ele in conversation_batches:
            batch_of_prompts.append(self.model.processor.apply_chat_template(ele, add_generation_prompt=True))
        
        inputs = self.model.processor(images=image_batches, text=batch_of_prompts, padding=True, return_tensors="pt").to(self.cfg.device)
        
        # print((inputs.input_ids != self.model.processor.tokenizer.pad_token_id).sum(dim=1))
        
        # output = self.model.model.generate(**inputs, max_new_tokens=100)
        # for n, p in self.model.model.named_parameters():
        #     if p.requires_grad:
        #         print(n, p.shape)
            
        activations = self.model.run_with_cache(
            list_of_hook_locations,
            **inputs,
        )[1][(block_layer, module_name)]
        

        
        if self.cfg.class_token:
          # Only keep the class token
          activations = activations[:,-3,:] 
          # activations = activations[:,0,:] # See the forward(), foward_head() methods of the VisionTransformer class in timm. 
          # Eg "x = x[:, 0]  # class token" - the [:,0] indexes the batch dimension then the token dimension

        return activations
    
    
    def get_sae_batches(self):
        # image_batches = self.get_image_batches()
        image_batches, conversation_batches = self.get_image_conversation_batches()
        max_batch_size = self.cfg.max_batch_size_for_vit_forward_pass
        number_of_mini_batches = len(image_batches) // max_batch_size
        remainder = len(image_batches) % max_batch_size
        sae_batches = []
        for mini_batch in trange(number_of_mini_batches, desc="Getting batches for SAE"):
            sae_batches.append(self.get_activations(image_batches[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size], conversation_batches[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size]))
        
        if remainder>0:
            sae_batches.append(self.get_activations(image_batches[-remainder:], conversation_batches[-remainder:]))
            
        sae_batches = torch.cat(sae_batches, dim = 0)
        sae_batches = sae_batches.to(self.cfg.device)
        return sae_batches
        

    def get_data_loader(self) -> DataLoader:
        '''
        Return a torch.utils.dataloader which you can get batches from.
        
        Should automatically refill the buffer when it gets to n % full. 
        (better mixing if you refill and shuffle regularly).
        
        '''
        batch_size = self.cfg.batch_size
        
        sae_batches = self.get_sae_batches()
        print(f"The actual amount of data loaded: {len(sae_batches)}")
        
        dataloader = iter(DataLoader(sae_batches, batch_size=batch_size, shuffle=True))
        # dataloader = DataLoader(sae_batches, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    
    def next_batch(self):
        """
        Get the next batch from the current DataLoader. 
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)
