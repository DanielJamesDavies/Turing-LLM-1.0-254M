import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
import json

from TuringLLM.inference_3 import TuringLLMForInference
from SAE_TopK import SAE



prompt = "Special Relativity"
max_tokens = 65
dataset_path = "../synthetic-dataset/auto_interp_dataset"



@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5



device = 'cuda' if torch.cuda.is_available() else 'cpu'



def run_prompt(prompt, max_tokens):
    torch.cuda.empty_cache()
    gc.collect()
    
    sae_dim = 10240
    sae_models = {}
    for layer_index in range(TuringLLMConfig.n_layer):
        k = 128 + (layer_index * 16)
        sae = SAE(TuringLLMConfig.n_embd, sae_dim, k).to(device)
        sae = torch.compile(sae)
        sae.load(f"sae/sae_layer_{layer_index}.pth")
        sae_models[str(layer_index)] = sae
    
    for i in range(TuringLLMConfig.n_layer+1):
        turing = TuringLLMForInference(sae_models=sae_models, replace_latents_layer_index=i-1)
        encoded_prompt = turing.tokenizer.encode(prompt)
        results = turing.generate_batch([encoded_prompt], max_length=max_tokens, tokenize=False, decode=False, ignore_end=True, get_logits=False)
        print("")
        if i == 0:
            print("No Layer Replacement")
        else:
            print(f"Layer Replaced: {i}")
        print("")
        print(results[0][1])
        print("")
        print("")



def run_shards():
    torch.cuda.empty_cache()
    gc.collect()
    
    sae_dim = 10240
    sae_models = {}
    for layer_index in range(TuringLLMConfig.n_layer):
        k = 128 + (layer_index * 16)
        sae = SAE(TuringLLMConfig.n_embd, sae_dim, k).to(device)
        sae = torch.compile(sae)
        sae.load(f"sae/sae_layer_{layer_index}.pth")
        sae_models[str(layer_index)] = sae
    
    losses = {}
    for i in range(TuringLLMConfig.n_layer):
        losses[i] = []
        
    max_examples = 128
    shard_index = 2000
    shard = np.load(f"{dataset_path}/shard_{shard_index}.npy")
    shard = np.split(shard, np.where(shard == -1)[0])
    shard = [sequence[sequence != -1] for sequence in shard if sequence.size > 1]
    
    torch.cuda.empty_cache()
    gc.collect()
    
    turing = TuringLLMForInference(collect_activations=False)
    
    for i in range(0, min(max_examples, len(shard))):
        start_time = time.time()
        
        print(f"Sequence {i+1} / {len(shard)}")
        
        original_probs = False
        for layer_index in range(TuringLLMConfig.n_layer+1):
            turing = TuringLLMForInference(return_logits=True, sae_models=sae_models, replace_latents_layer_index=layer_index-1)
            results, logits = turing.generate_batch([shard[i].tolist()], max_length=max_tokens, tokenize=False, decode=False, ignore_end=True)
            if layer_index == 0:
                original_probs = F.softmax(logits, dim=-1)
            else:
                criterion = nn.KLDivLoss(reduction='batchmean')
                probs = F.log_softmax(logits, dim=-1)
                loss = criterion(probs, original_probs)
                print(f"  Layer {str(layer_index).zfill(len(str(TuringLLMConfig.n_layer)))} KL Divergence: {loss.item():.6f}")
                losses[layer_index-1].append(loss.item())
                
        with open('replaced_latents_losses.json', 'w', encoding='utf-8') as f:
            json.dump(losses, f)
                        
        print(f"Duration: {time.time()-start_time:.2f}s")
        print("")

    turing.clear()
    del turing



# run_prompt(prompt, max_tokens)
run_shards()
