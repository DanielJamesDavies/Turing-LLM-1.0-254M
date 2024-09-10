import os
import time
import numpy as np
import torch
import logging
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


explanation_length = 20



logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", use_fast=True, local_files_only=True, _fast_init=True)
except:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", use_fast=True, _fast_init=True)



explanation_scores_path = "scope_scores"
os.makedirs(explanation_scores_path, exist_ok=True)
explanations_path = "explanations"
latents_path = "latents"
latents_filenames_list = [
    {
        "type": "from_sequence",
        "tokens": "latents_avg_mlp_down_tokens_from_sequence.npy",
        "values": "latents_avg_mlp_down_values_from_sequence.pth",
        "explanations": "latents_avg_mlp_down_explanations_from_sequence.npy",
        "save": "latents_avg_mlp_down_explanation_scores_from_sequence.npy"
    },
    {
        "type": "from_sequence",
        "tokens": "latents_avg_sae_tokens_from_sequence.npy",
        "values": "latents_avg_sae_values_from_sequence.pth",
        "explanations": "latents_avg_sae_explanations_from_sequence.npy",
        "save": "latents_avg_sae_explanation_scores_from_sequence.npy"
    },
    {
        "type": "from_prev_latents",
        "tokens": "latents_mlp_down_latents_from_prev_latents.npy",
        "values": "latents_mlp_down_values_from_prev_latents.pth",
        "explanations": "latents_avg_mlp_down_explanations_from_prev_latents.npy",
        "save": "latents_avg_mlp_down_explanation_scores_from_prev_latents.npy"
    },
    {
        "type": "from_next_latents",
        "tokens": "latents_mlp_down_latents_from_next_latents.npy",
        "values": "latents_mlp_down_values_from_next_latents.pth",
        "explanations": "latents_avg_mlp_down_explanations_from_next_latents.npy",
        "save": "latents_avg_mlp_down_explanation_scores_from_next_latents.npy"
    },
    {
        "type": "from_prev_latents",
        "tokens": "latents_sae_latents_from_prev_latents.npy",
        "values": "latents_sae_values_from_prev_latents.pth",
        "explanations": "latents_avg_sae_explanations_from_prev_latents.npy",
        "save": "latents_avg_sae_explanation_scores_from_prev_latents.npy"
    },
    {
        "type": "from_sequence",
        "tokens": "latents_avg_residuals_tokens_from_sequence.npy",
        "values": "latents_avg_residuals_values_from_sequence.pth",
        "explanations": "latents_avg_residuals_explanations_from_sequence.npy",
        "save": "latents_avg_residuals_explanation_scores_from_sequence.npy"
    },
    {
        "type": "from_prev_latents",
        "tokens": "latents_residuals_latents_from_prev_latents.npy",
        "values": "latents_residuals_values_from_prev_latents.pth",
        "explanations": "latents_avg_residuals_explanations_from_prev_latents.npy",
        "save": "latents_avg_residuals_explanation_scores_from_prev_latents.npy"
    },
    {
        "type": "from_next_latents",
        "tokens": "latents_residuals_latents_from_next_latents.npy",
        "values": "latents_residuals_values_from_next_latents.pth",
        "explanations": "latents_avg_residuals_explanations_from_next_latents.npy",
        "save": "latents_avg_residuals_explanation_scores_from_next_latents.npy"
    }
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

text_encoder_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
text_encoder_model = text_encoder_model.to(device)

for latents_filenames in latents_filenames_list:
    print("")
    print("")
    print("")
    print("")
    print("")
    print("Building " + latents_filenames["save"])
    print("")
    
    latents_tokens = np.load(latents_path + "/" + latents_filenames['tokens'])
    latents_values = torch.load(latents_path + "/" + latents_filenames['values'])
    
    latent_range = int(latents_tokens.shape[1] * 0.25)
        
    explanations = np.load(explanations_path + "/" + latents_filenames['explanations'])
    explanation_scores = np.zeros((latents_tokens.shape[0], latents_tokens.shape[1]))
    
    layers_range = range(latents_tokens.shape[0])
        
    layer_averages = []
    
    for layer_index in layers_range:
        print(f"Layer {layer_index+1} / {latents_tokens.shape[0]}", end="   ")
        
        start_time = time.time()
        
        layer_explanations = [tokenizer.decode(explanations[layer_index][i]).lstrip("<|endoftext|><s>").strip().replace("<|endoftext|>", "<split_in_text>").replace("<s>", "") for i in range(latents_tokens.shape[1])]
        layer_explanations_embeddings = text_encoder_model.encode(layer_explanations, device=device)
        layer_explanations_cosine_similarities = cosine_similarity(layer_explanations_embeddings, layer_explanations_embeddings)
        
        layer_explanation_scores = [0 for latent_index in range(latents_tokens.shape[1])]
        
        for latent_index in range(latents_tokens.shape[1]):
            latent_layer_explanations_cosine_similarities = [cos_sim for i, cos_sim in enumerate(layer_explanations_cosine_similarities[latent_index]) if i != latent_index]
            latent_layer_explanations_cosine_similarity_average = sum(latent_layer_explanations_cosine_similarities) / len(latent_layer_explanations_cosine_similarities)
            layer_explanation_scores[latent_index] = latent_layer_explanations_cosine_similarity_average
        
        explanation_scores[layer_index] = np.array(layer_explanation_scores)
        
        layer_average = sum(layer_explanation_scores) / len(layer_explanation_scores)
        layer_averages.append(layer_average)
        print(f"Layer Average: {layer_average:.3f}", end="   ")
            
        print(f"Duration: {time.time()-start_time:.2f}s")

    layers_average = sum(layer_averages) / len(layer_averages)
    print("")
    print(f"Average: {layers_average:.3f}")
    
    np.save(explanation_scores_path  + "/" + latents_filenames["save"], explanation_scores)

    print("")
    print("Saved " + latents_filenames["save"])
