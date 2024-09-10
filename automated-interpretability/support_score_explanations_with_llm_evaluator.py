import os
import time
import numpy as np
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc
from sklearn.metrics.pairwise import cosine_similarity


explanation_scores_length = 2
explanation_length = 20



logging.getLogger("transformers").setLevel(logging.ERROR)

class LLM:
    def __init__(self, model_id = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_id = model_id
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True, _fast_init=True, attn_implementation="flash_attention_2")
        except:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, _fast_init=True, attn_implementation="flash_attention_2")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, local_files_only=True, _fast_init=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, _fast_init=True)
            
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        
        self.device = torch.cuda.current_device()
        
        self.model.eval()

    def generate_batch(self, batch, max_length=4096, skip_special_tokens=True, reset_on_low_tok_sec=False):
        new_batch = []
        max_input_token_length = 0
        for i in range(len(batch)):
            new_batch.append(self.tokenizer.apply_chat_template(batch[i], tokenize=False, add_generation_prompt=True) + "Connection Score (0-3):")
            tokens = self.tokenizer.apply_chat_template(batch[i], tokenize=True, add_generation_prompt=True, return_tensors="pt")
            max_input_token_length = max(max_input_token_length, tokens.shape[1])
        
        def get_messages():
            for messages in new_batch:
                yield messages
        
        generation_args = {
            "batch_size": 21,
            "return_tensors": True,
            "max_new_tokens": min(max_length, self.tokenizer.model_max_length - max_input_token_length),
        }
        
        outputs = []
        with torch.no_grad():
            for out in self.pipe(get_messages(), **generation_args):
                outputs.append(out[0]["generated_token_ids"])
                if len(outputs) % generation_args["batch_size"] == 0:
                    torch.cuda.empty_cache()
        
        def get_last_assistant_token_index(tokens):
            for i in range(len(tokens)-1,-1,-1):
                if tokens[i] == 32001:
                    return i
        
        generated_texts = []
        for output in outputs:
            last_assistant_token_index = get_last_assistant_token_index(output)
            generated_tokens = output[last_assistant_token_index+1:]
            decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=skip_special_tokens)
            generated_texts.append(decoded_output)

        return generated_texts

    def cleanup(self):
        print("Cleaning up model...")
        del self.model
        del self.tokenizer
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()
        
    def reset(self):
        self.cleanup()
        self.setup_model()
        
        
llm = LLM()



explanation_score_prompt = """
Please respond with a score from 0-5 of how well the topic describes the text. A higher score would mean a connection between the topic and text, whereas a low score would mean no or little connection between the two.

START OF TEXT
Text: <|text|>
END OF TEXT

START OF TOPIC
Topic: <|explanation|>
END OF TOPIC

Please respond with "Connection Score (0-5): " and then the score number from 0-5 on one line. Thank you so much!

How to choose score:
- 0 - no relationship.
- 1 - extremely weak relationship.
- 2 - a very weak similarity between the topic and the text.
- 3 - a similarity between the topic and the text.
- 4 - a medium connection.
- 5 - a strong connection.

Aim to give higher scores.

From 0 to 5, is the text related to the topic?
"""



explanation_scores_path = "llm_evaluator_support_scores"
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
        "save": "latents_avg_sae_explanation_scores_from_sequence_similar_explanation.npy"
    }
]

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
    explanation_scores = np.zeros((latents_tokens.shape[0], latent_range, 6), dtype=np.uint16)
    if "end_explanations" in latents_filenames:
        end_explanations = np.load(explanations_path + "/" + latents_filenames['end_explanations'])
    
    layers_range = range(latents_tokens.shape[0])
    if latents_filenames["type"] == "from_next_latents":
        layers_range = layers_range[::-1]
        
    for layer_index in layers_range:
        print(f"Layer {layer_index+1} / {latents_tokens.shape[0]}")
        prompts = []
        prompt_indices = []
        
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
        
        if "get_similar_explanation" in latents_filenames and latents_filenames["get_similar_explanation"] is True:
            sae_weights = torch.load(f"SAE/sae/sae_layer_{layer_index}.pth")
            sae_direction_cosine_similarities = cosine_similarity(sae_weights["encoder.weight"].cpu(), sae_weights["encoder.weight"].cpu())
        
        layer_explanations = [llm.tokenizer.decode(explanations[layer_index][i]).lstrip("<|endoftext|><s>").strip().replace("<|endoftext|>", "<split_in_text>").replace("<s>", "") for i in range(latents_tokens.shape[1])]
        
        for latent_index in range(latent_range):
            sequences = []
            
            explanation = layer_explanations[latent_index]
            
            if "get_similar_explanation" in latents_filenames and latents_filenames["get_similar_explanation"] is True:
                latent_sae_direction_cosine_similarities = [{ "latent_index": i, "cos_sim": cos_sim } for i, cos_sim in enumerate(sae_direction_cosine_similarities[latent_index]) if i != latent_index]
                latent_latents_cosine_similarities = list(sorted(latent_sae_direction_cosine_similarities, key=lambda x: x['cos_sim'], reverse=True))
                explanation = layer_explanations[latent_latents_cosine_similarities[0]["latent_index"]].strip()
            
            if latents_filenames["type"] == "from_sequence":
                sequences = [llm.tokenizer.decode(latents_tokens[layer_index][latent_index][(i*2)+1]) for i in range(6)]
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
                prev_latent_explanations = [llm.tokenizer.decode(prev_latent_explanation).lstrip("<|endoftext|><s>").strip() for prev_latent_explanation in prev_latent_explanations]
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
                next_latent_explanations = [llm.tokenizer.decode(next_latent_explanation).lstrip("<|endoftext|><s>").strip() for next_latent_explanation in next_latent_explanations]
                sequences = list(next_latent_explanations)
            
            for sequence_index, sequence in enumerate(sequences):
                if sequence_index < 6:
                    prompt = str(explanation_score_prompt).strip()
                    prompt = prompt.replace("<|explanation|>", explanation)
                    prompt = prompt.replace("<|text|>", sequence)
                    prompt = [{ "role": "user", "content": prompt.strip() }]
                    prompts.append(prompt)
                    prompt_indices.append([layer_index, latent_index, sequence_index])

        start_time = time.time()
        
        response = llm.generate_batch(prompts, max_length = explanation_length)

        layer_explanation_scores = [[500, 500, 500, 500, 500, 500] for _ in range(latent_range)]
        for i, explanation_score in enumerate(response):
            try:
                explanation_score = int(explanation_score[len("Relationship Percentage:"):].strip().split("\n")[0].strip()[0])
            except:
                explanation_score = 500
                
            layer_explanation_scores[prompt_indices[i][1]][prompt_indices[i][2]] = explanation_score
                
            if i < 12:
                print("")
                print("Latent", prompt_indices[i][1], " Sequence", prompt_indices[i][2])
                print("")
                print("Prompt:", prompts[i][0]["content"])
                print("")
                print("Explanation Score:", explanation_score)
                print("")
                print("")
        
        explanation_scores[layer_index] = np.array(layer_explanation_scores, dtype=np.uint16)
            
        print(f"Duration: {time.time()-start_time:.2f}s")
        print("")
        print("")
        print("")
        print("")

    np.save(explanation_scores_path  + "/" + latents_filenames["save"], explanation_scores)

    print("Saved " + latents_filenames["save"])
