from functools import partial

import numpy as np
import plotly_express as px
import torch
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

import wandb
from sae_training.activations_store import ActivationsStore
from sae_training.vit_activations_store import ViTActivationsStore
from sae_training.optim import get_scheduler
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.config import ViTSAERunnerConfig
from PIL import Image


def train_sae_on_vision_transformer(
    model: HookedVisionTransformer,
    sparse_autoencoder: SparseAutoencoder,
    activation_store: ViTActivationsStore,
):
    feature_sampling_method = sparse_autoencoder.cfg.feature_sampling_method
    batch_size = sparse_autoencoder.cfg.batch_size
    total_training_tokens = sparse_autoencoder.cfg.total_training_tokens
    n_checkpoints = sparse_autoencoder.cfg.n_checkpoints
    
    if sparse_autoencoder.cfg.log_to_wandb:
        wandb.init(project="mats-hugo")
    if feature_sampling_method is not None:
        feature_sampling_method = feature_sampling_method.lower()
    
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // n_checkpoints))[1:]
    print(f"checkpoint_thresholds: {checkpoint_thresholds}")
    
    # track active features
    act_freq_scores = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
    n_forward_passes_since_fired = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
    n_frac_active_tokens = 0
    
    optimizer = Adam(sparse_autoencoder.parameters(), lr = sparse_autoencoder.cfg.lr)
    scheduler = get_scheduler(
        sparse_autoencoder.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = sparse_autoencoder.cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=sparse_autoencoder.cfg.lr / 10, # heuristic for now. 
    )
    sparse_autoencoder.initialize_b_dec(activation_store)
    sparse_autoencoder.train()
    

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        # print(f"n_training_steps:{n_training_steps}")
        # Do a training step.
        sparse_autoencoder.train()
        # Make sure the W_dec is still zero-norm
        sparse_autoencoder.set_decoder_norm_to_unit_norm()
            
        # after resampling, reset the sparsity:
        if (n_training_steps + 1) % sparse_autoencoder.cfg.feature_sampling_window == 0: # feature_sampling_window divides dead_sampling_window
            
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()

            if sparse_autoencoder.cfg.log_to_wandb:
                wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                wandb.log(
                    {   
                        "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                        "plots/feature_density_line_chart": wandb_histogram,
                    },
                    step=n_training_steps,
                )
            
            act_freq_scores = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
            n_frac_active_tokens = 0


        scheduler.step()
        optimizer.zero_grad()
        
        ghost_grad_neuron_mask = (n_forward_passes_since_fired > sparse_autoencoder.cfg.dead_feature_window).bool()
        sae_in = activation_store.next_batch()
        
        # Forward and Backward Passes
        sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sparse_autoencoder(
            sae_in,
            ghost_grad_neuron_mask,
        )
        
        did_fire = ((feature_acts > 0).float().sum(-2) > 0)
        n_forward_passes_since_fired += 1
        n_forward_passes_since_fired[did_fire] = 0
        
        n_training_tokens += batch_size

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            n_frac_active_tokens += batch_size
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()  # 增加的一句话

            if sparse_autoencoder.cfg.log_to_wandb and ((n_training_steps + 1) % sparse_autoencoder.cfg.wandb_log_frequency == 0):
                # metrics for currents acts
                l0 = (feature_acts > 0).float().sum(-1).mean()
                current_learning_rate = optimizer.param_groups[0]["lr"]
                
                per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                total_variance = sae_in.pow(2).sum(-1)
                explained_variance = 1 - per_token_l2_loss/total_variance
                
                wandb.log(
                    {
                        # losses
                        "losses/mse_loss": mse_loss.item(),
                        "losses/l1_loss": l1_loss.item() / sparse_autoencoder.l1_coefficient, # normalize by l1 coefficient
                        "losses/ghost_grad_loss": ghost_grad_loss.item(),
                        "losses/overall_loss": loss.item(),
                        # variance explained
                        "metrics/explained_variance": explained_variance.mean().item(),
                        "metrics/explained_variance_std": explained_variance.std().item(),
                        "metrics/l0": l0.item(),
                        # sparsity
                        "sparsity/mean_passes_since_fired": n_forward_passes_since_fired.mean().item(),
                        "sparsity/n_passes_since_fired_over_threshold": ghost_grad_neuron_mask.sum().item(),
                        "sparsity/below_1e-5": (feature_sparsity < 1e-5)
                        .float()
                        .mean()
                        .item(),
                        "sparsity/below_1e-6": (feature_sparsity < 1e-6)
                        .float()
                        .mean()
                        .item(),
                        "sparsity/dead_features": (
                            feature_sparsity < sparse_autoencoder.cfg.dead_feature_threshold
                        )
                        .float()
                        .mean()
                        .item(),
                        "details/n_training_tokens": n_training_tokens,
                        "details/current_learning_rate": current_learning_rate,
                    },
                    step=n_training_steps,
                )
                
            if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
                sparse_autoencoder.eval()
                image_file = "image1.jpg"
                raw_image = Image.open(image_file)
                
                conversation = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Please describe this image."}
                ]}]
                
                prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
                model_inputs = model.processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
                input_ids = model_inputs.input_ids
                attention_mask = model_inputs.attention_mask
                pixel_values = model_inputs.pixel_values
                aspect_ratio_ids = model_inputs.aspect_ratio_ids  # for Llama-3.2-11B-Vision-Instruct
                aspect_ratio_mask = model_inputs.aspect_ratio_mask
                cross_attention_mask = model_inputs.cross_attention_mask
                generated_ids = input_ids.clone()
                
                def sae_hook1(activations):
                    activations[:,-1,:] = sparse_autoencoder(activations[:,-1,:])[0]
                    # activations[:,-1,:] = activations[:,-1,:]
                    # print(activations.shape)
                    return (activations,)
                        
                sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook1, return_module_output=True)] 
                max_token = 100
                print("test case:")
                for ele in range(max_token):
                    # print(ele)
                    outputs = model.run_with_hooks(
                        sae_hooks,
                        return_type='output',
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        aspect_ratio_ids=aspect_ratio_ids,
                        aspect_ratio_mask=aspect_ratio_mask,
                        cross_attention_mask=cross_attention_mask,
                        # image_sizes=image_sizes,
                    )
                    logits = outputs.logits[:, -1, :]  
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    new_mask = torch.ones((attention_mask.shape[0], 1), device=sparse_autoencoder.cfg.device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
                    if cross_attention_mask is not None:
                        size_0, size_1, size_2, size_3 = cross_attention_mask.shape
                        new_cross_attention_mask = torch.ones(
                            (size_0, 1, size_2, size_3),
                            device=cross_attention_mask.device
                        )
                        cross_attention_mask = torch.cat([cross_attention_mask, new_cross_attention_mask], dim=1)

                    torch.cuda.empty_cache()
                output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                print(output_texts)
                sparse_autoencoder.train()

            pbar.set_description(
                f"{n_training_steps}| Loss {loss.item():.3f} | MSE Loss {mse_loss.item():.3f} | L1 {l1_loss.item():.3f} | Ghost Grad Loss {ghost_grad_loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        


        # checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            cfg = sparse_autoencoder.cfg
            path = f"{sparse_autoencoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}.pt"
            log_feature_sparsity_path = f"{sparse_autoencoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}_log_feature_sparsity.pt"
            sparse_autoencoder.save_model(path)
            torch.save(log_feature_sparsity, log_feature_sparsity_path)
            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0
            if cfg.log_to_wandb:
                model_artifact = wandb.Artifact(
                    f"{sparse_autoencoder.get_name()}", type="model", metadata=dict(cfg.__dict__)
                )
                model_artifact.add_file(path)
                wandb.log_artifact(model_artifact)
                
                sparsity_artifact = wandb.Artifact(
                    f"{sparse_autoencoder.get_name()}_log_feature_sparsity", type="log_feature_sparsity", metadata=dict(cfg.__dict__)
                )
                sparsity_artifact.add_file(log_feature_sparsity_path)
                wandb.log_artifact(sparsity_artifact)
  
                
                   
        n_training_steps += 1
        
        
    if n_checkpoints > 0:
        log_feature_sparsity_path = f"{sparse_autoencoder.cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}_log_feature_sparsity.pt"
        sparse_autoencoder.save_model(path)
        # torch.save(log_feature_sparsity, log_feature_sparsity_path)
        # if cfg.log_to_wandb:
        #     sparsity_artifact = wandb.Artifact(
        #             f"{sparse_autoencoder.get_name()}_log_feature_sparsity", type="log_feature_sparsity", metadata=dict(cfg.__dict__)
        #         )
        #     sparsity_artifact.add_file(log_feature_sparsity_path)
        #     wandb.log_artifact(sparsity_artifact)
        

    return sparse_autoencoder


@torch.no_grad()
def run_evals(sparse_autoencoder: SparseAutoencoder, activation_store: ViTActivationsStore, model: HookedVisionTransformer, n_training_steps: int):
    def zero_ablation(activations):
        activations[:,0,:] = torch.zeros_like(activations[:,0,:]).to(activations.device)
        return (activations,) # activations of size [batch, token, dimension]
    
    def sae_hook(activations):
        activations[:,0,:] = sparse_autoencoder(activations[:,0,:])[0]
        return (activations,)
    
    model_inputs = activation_store.get_batch_of_images_and_labels()
    original_loss = model(return_type='loss', **model_inputs).item()
    sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)]  # False
    reconstruction_loss = model.run_with_hooks(sae_hooks, return_type='loss', **model_inputs)  # loss
    zero_ablation_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, zero_ablation, return_module_output=True)]
    zero_ablation_loss = model.run_with_hooks(zero_ablation_hooks, return_type='loss', **model_inputs).item() # loss
    
    reconstruction_score = (reconstruction_loss - original_loss)/(zero_ablation_loss-original_loss)
    
    if sparse_autoencoder.cfg.log_to_wandb:
        wandb.log(
            {   
                # Contrastive Loss
                "metrics/contrastive_loss_score": reconstruction_score,
                "metrics/original_contrastive_loss": original_loss,
                "metrics/contrastive_loss_with_sae": reconstruction_loss,
                "metrics/contrastive_loss_with_ablation": zero_ablation_loss,
                
            },
            step=n_training_steps,
        )


def kl_divergence_attention(y_true, y_pred):

    # Compute log probabilities for KL divergence
    log_y_true = torch.log2(y_true + 1e-10)
    log_y_pred = torch.log2(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)