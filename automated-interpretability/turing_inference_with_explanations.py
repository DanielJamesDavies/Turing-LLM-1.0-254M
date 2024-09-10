from dataclasses import dataclass
import numpy as np
import torch
import torch.cuda as cuda
import gc

from TuringLLM.inference import TuringLLMForInference
from SAE.SAE_TopK import SAE



prompt = "Special Relativity"
max_tokens = 65
explaining_token_index = 21
explanation_filename = "latents_avg_sae_explanations_from_sequence.npy"
use_saes = True



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



def run(prompt, explanations, use_saes, max_tokens, explaining_token_index):
    torch.cuda.empty_cache()
    gc.collect()
    
    turing = TuringLLMForInference(collect_latents=True, max_length=max_tokens)
    encoded_prompt = turing.tokenizer.encode(prompt)
    results, latents = turing.generate_batch([encoded_prompt], max_length=max_tokens, tokenize=False, decode=True, ignore_end=True)
    print("")
    print("")
    
    
    # Print Tokens
    remaining_result_string = results[0][1]
    for i, token in enumerate(results[0][0]):
        if i == explaining_token_index:
            print("\033[0m", end="")
        elif i % 2 == 0:
            print("\033[35m", end="")
        else:
            print("\033[36m", end="")
        decoded_token = turing.tokenizer.decode([token])
        for character in remaining_result_string:
            if remaining_result_string[:len(decoded_token)] == decoded_token:
                print(decoded_token, end="")
                remaining_result_string = remaining_result_string[len(decoded_token):]
                break
            print(character, end="")
            remaining_result_string = remaining_result_string[1:]
    print("\033[0m")
    
    
    # Get Latents
    latents = latents[0]
    if use_saes is True:
        sae_dim = 10240
        sae_models = {}
        for layer_index in range(TuringLLMConfig.n_layer):
            k = 128 + (layer_index * 16)
            sae = SAE(TuringLLMConfig.n_embd, sae_dim, k, only_encoder=True).to(device)
            sae = torch.compile(sae)
            sae.load(f"SAE/sae/sae_layer_{layer_index}.pth")
            sae_models[str(layer_index)] = sae
            
        sae_latents = {}
        cuda_streams = {}
        for layer_index in range(TuringLLMConfig.n_layer):
            cuda_stream = cuda.Stream()
            cuda_streams[str(layer_index)] = cuda_stream
            
            with cuda.stream(cuda_stream):
                layer_sae_latents, _, _ = sae_models[str(layer_index)].encode(latents[layer_index][:, explaining_token_index, :].squeeze(0).to(device))
                layer_sae_latents = layer_sae_latents.detach().cpu()
                sae_latents[str(layer_index)] = layer_sae_latents
                
        for cuda_stream in cuda_streams.values():
            cuda_stream.synchronize()
        
        latents = [sae_latents[str(layer_index)] for layer_index in range(TuringLLMConfig.n_layer)]
    
    
    
    # Get Explanations of Top Activated Latents
    for layer_index, layer_new_latents in enumerate(latents):
        if "_from_prev_latents" in explanation_filename and layer_index == 0:
            continue
        if "_from_next_latents" in explanation_filename and layer_index == len(latents) - 1:
            continue
        
        print("")
        print("")
        print(f"Layer {layer_index+1}")
        
        if use_saes is True:
            latent_values = layer_new_latents
        else:
            latent_values = layer_new_latents[:, explaining_token_index, :].squeeze(0)
            
        ordered_indices = torch.argsort(latent_values, descending=True).tolist()
        
        for latent_index in ordered_indices[:4]:
            if "_from_prev_latents" in explanation_filename:
                explanation_tokens = [token for token in explanations[layer_index-1][latent_index] if token != 1]
            else:
                explanation_tokens = [token for token in explanations[layer_index][latent_index] if token != 1]
            explanation = turing.tokenizer.decode(explanation_tokens)
            print(f"Absolute Value: {latent_values[latent_index].item():.3f}", end="  |  ")
            print(f"Explanation: {explanation}")
        print("")



explanations = np.load("explanations/" + explanation_filename)
run(prompt, explanations, use_saes, max_tokens, explaining_token_index)
