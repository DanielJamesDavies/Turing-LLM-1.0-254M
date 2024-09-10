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



explanation_scores_path = "cos_sim_evaluator_support_scores"
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
        "end_explanations": "latents_avg_mlp_down_explanations_from_sequence.npy",
        "explanations": "latents_avg_mlp_down_explanations_from_prev_latents.npy",
        "save": "latents_avg_mlp_down_explanation_scores_from_prev_latents.npy"
    },
    {
        "type": "from_next_latents",
        "tokens": "latents_mlp_down_latents_from_next_latents.npy",
        "values": "latents_mlp_down_values_from_next_latents.pth",
        "end_explanations": "latents_avg_mlp_down_explanations_from_sequence.npy",
        "explanations": "latents_avg_mlp_down_explanations_from_next_latents.npy",
        "save": "latents_avg_mlp_down_explanation_scores_from_next_latents.npy"
    },
    {
        "type": "from_prev_latents",
        "tokens": "latents_sae_latents_from_prev_latents.npy",
        "values": "latents_sae_values_from_prev_latents.pth",
        "end_explanations": "latents_avg_sae_explanations_from_sequence.npy",
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
        "end_explanations": "latents_avg_residuals_explanations_from_sequence.npy",
        "explanations": "latents_avg_residuals_explanations_from_prev_latents.npy",
        "save": "latents_avg_residuals_explanation_scores_from_prev_latents.npy"
    },
    {
        "type": "from_next_latents",
        "tokens": "latents_residuals_latents_from_next_latents.npy",
        "values": "latents_residuals_values_from_next_latents.pth",
        "end_explanations": "latents_avg_residuals_explanations_from_sequence.npy",
        "explanations": "latents_avg_residuals_explanations_from_next_latents.npy",
        "save": "latents_avg_residuals_explanation_scores_from_next_latents.npy"
    },
    {
        "type": "from_sequence",
        "get_similar_explanation": True,
        "tokens": "latents_avg_sae_tokens_from_sequence.npy",
        "values": "latents_avg_sae_values_from_sequence.pth",
        "explanations": "latents_avg_sae_explanations_from_sequence.npy",
        "save": "latents_avg_sae_explanation_scores_from_sequence_with_similar_explanation.npy"
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
    
    explanations = np.load(explanations_path + "/" + latents_filenames['explanations'])
    explanation_scores = np.zeros((latents_tokens.shape[0], latents_tokens.shape[1], 6))
    if "end_explanations" in latents_filenames:
        end_explanations = np.load(explanations_path + "/" + latents_filenames['end_explanations'])
    
    layers_average_explanation_scores = []
      
    for layer_index in range(latents_tokens.shape[0]):
        print(f"Layer {layer_index+1} / {latents_tokens.shape[0]}", end="  ")
        layer_explanation_scores = [[-1, -1, -1, -1, -1, -1] for latent_index in range(latents_tokens.shape[1])]
        layer_average_explanation_scores = [-1 for latent_index in range(latents_tokens.shape[1])]
        start_time = time.time()
        
        if "get_similar_explanation" in latents_filenames and latents_filenames["get_similar_explanation"] is True:
            sae_weights = torch.load(f"SAE/sae/sae_layer_{layer_index}.pth")
            sae_direction_cosine_similarities = cosine_similarity(sae_weights["encoder.weight"].cpu(), sae_weights["encoder.weight"].cpu())
        
        latents_indices_frequencies = {}
        if latents_filenames["type"] == "from_prev_latents" or latents_filenames["type"] == "from_next_latents":
            for latent_index in range(latents_tokens.shape[1]):
                for i in range(12):
                    for curr_latent_index in latents_tokens[layer_index][latent_index][i]:
                        if str(curr_latent_index) not in latents_indices_frequencies:
                            latents_indices_frequencies[str(curr_latent_index)] = 1
                        else:
                            latents_indices_frequencies[str(curr_latent_index)] += 1
        max_latent_frequency = int(latents_tokens.shape[1] * 0.25)
        
        layer_explanations = [tokenizer.decode(explanations[layer_index][i]).lstrip("<|endoftext|><s>").strip().replace("<|endoftext|>", "<split_in_text>").replace("<s>", "") for i in range(latents_tokens.shape[1])]
        layer_explanations_embeddings = text_encoder_model.encode(layer_explanations, device=device)
        
        for latent_index in range(latents_tokens.shape[1]):
            sequences = []
            
            if "get_similar_explanation" in latents_filenames and latents_filenames["get_similar_explanation"] is True:
                latent_sae_direction_cosine_similarities = [{ "latent_index": i, "cos_sim": cos_sim } for i, cos_sim in enumerate(sae_direction_cosine_similarities[latent_index]) if i != latent_index]
                latent_latents_cosine_similarities = list(sorted(latent_sae_direction_cosine_similarities, key=lambda x: x['cos_sim'], reverse=True))
                similar_explanation = layer_explanations_embeddings[latent_latents_cosine_similarities[0]["latent_index"]]
            
            if latents_filenames["type"] == "from_sequence":
                sequences = [tokenizer.decode(latents_tokens[layer_index][latent_index][(i*2)+1]) for i in range(6)]
                sequences = [sequence.lstrip("<|endoftext|><s>").strip().replace("<|endoftext|>", "<split_in_text>").replace("<s>", "") for i, sequence in enumerate(sequences)]

            elif latents_filenames["type"] == "from_prev_latents":
                prev_latent_indices_frequencies = {}
                for i in range(4):
                    for prev_latent_index in latents_tokens[layer_index][latent_index][(i*2)+1]:
                        if latents_indices_frequencies[str(prev_latent_index)] < max_latent_frequency:
                            if str(prev_latent_index) not in prev_latent_indices_frequencies:
                                prev_latent_indices_frequencies[str(prev_latent_index)] = 1
                            else:
                                prev_latent_indices_frequencies[str(prev_latent_index)] += 1
                prev_latent_indices = sorted(prev_latent_indices_frequencies, key=prev_latent_indices_frequencies.get, reverse=True)[:11]
                
                if layer_index == 0:
                    prev_latent_explanations = [end_explanations[0][int(prev_latent_index)] for prev_latent_index in prev_latent_indices]
                else:
                    prev_latent_explanations = [explanations[layer_index-1][int(prev_latent_index)] for prev_latent_index in prev_latent_indices]
                    
                prev_latent_explanations = [[x for x in prev_latent_explanation if x != 1] for prev_latent_explanation in prev_latent_explanations]
                prev_latent_explanations = [tokenizer.decode(prev_latent_explanation).lstrip("<|endoftext|><s>").strip() for prev_latent_explanation in prev_latent_explanations]
                sequences = list(prev_latent_explanations)

            elif latents_filenames["type"] == "from_next_latents":
                next_latent_indices_frequencies = {}
                for i in range(4):
                    for next_latent_index in latents_tokens[layer_index][latent_index][(i*2)+1]:
                        if latents_indices_frequencies[str(next_latent_index)] < max_latent_frequency:
                            if str(next_latent_index) not in next_latent_indices_frequencies:
                                next_latent_indices_frequencies[str(next_latent_index)] = 1
                            else:
                                next_latent_indices_frequencies[str(next_latent_index)] += 1
                next_latent_indices = sorted(next_latent_indices_frequencies, key=next_latent_indices_frequencies.get, reverse=True)[:11]
                    
                if layer_index == latents_tokens.shape[0] - 1:
                    next_latent_explanations = [end_explanations[-1][int(next_latent_index)] for next_latent_index in next_latent_indices]
                else:
                    next_latent_explanations = [explanations[layer_index+1][int(next_latent_index)] for next_latent_index in next_latent_indices]
                    
                next_latent_explanations = [[x for x in next_latent_explanation if x != 1] for next_latent_explanation in next_latent_explanations]
                next_latent_explanations = [tokenizer.decode(next_latent_explanation).lstrip("<|endoftext|><s>").strip() for next_latent_explanation in next_latent_explanations]
                sequences = list(next_latent_explanations)
                
                
                
            explanation = layer_explanations_embeddings[latent_index]
            
            if len(sequences) > 6:
                sequences = sequences[:6]
            
            if len(sequences) != 0:
                sequences_embeddings = text_encoder_model.encode(sequences, device=device)
                
                cosine_similarities = cosine_similarity([explanation], sequences_embeddings)[0]
                    
                if "get_similar_explanation" in latents_filenames and latents_filenames["get_similar_explanation"] is True:
                    similar_cosine_similarities = cosine_similarity([similar_explanation], sequences_embeddings)[0]
                    for i in range(len(cosine_similarities)):
                        if similar_cosine_similarities[i] > cosine_similarities[i]:
                            cosine_similarities[i] = similar_cosine_similarities[i]
                        
                for i, cos_sim in enumerate(cosine_similarities):
                    layer_explanation_scores[latent_index][i] = cos_sim
                
                layer_average_explanation_scores[latent_index] = sum(cosine_similarities) / len(cosine_similarities)
        
        explanation_scores[layer_index] = np.array(layer_explanation_scores)
        
        layer_average_explanation_scores = [score for score in layer_average_explanation_scores if score != -1]
        layer_average_explanation_score = sum(layer_average_explanation_scores) / len(layer_average_explanation_scores)
        layers_average_explanation_scores.append(layer_average_explanation_score)
        print(f"Layer Average: {layer_average_explanation_score:.3f}", end="  ")
        
        print(f"Duration: {time.time()-start_time:.2f}s")
    
    np.save(explanation_scores_path  + "/" + latents_filenames["save"], explanation_scores)

    layers_average_explanation_score = sum(layers_average_explanation_scores) / len(layers_average_explanation_scores)
    print("")
    print(f"Average: {layers_average_explanation_score:.3f}", end="  ")

    print("")
    print("Saved " + latents_filenames["save"])
