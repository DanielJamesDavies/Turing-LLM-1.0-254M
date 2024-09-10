import os
import time
import numpy as np
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc



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
            new_batch.append(self.tokenizer.apply_chat_template(batch[i], tokenize=False, add_generation_prompt=True) + "TOPIC(S) COMMON IN ALL TEXTS:")
            tokens = self.tokenizer.apply_chat_template(batch[i], tokenize=True, add_generation_prompt=True, return_tensors="pt")
            max_input_token_length = max(max_input_token_length, tokens.shape[1])
        
        def get_messages():
            for messages in new_batch:
                yield messages
        
        generation_args = {
            "batch_size": 8,
            "return_tensors": True,
            "max_new_tokens": min(max_length, self.tokenizer.model_max_length - max_input_token_length),
            "do_sample": True,
            "top_k": 20
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



explanation_prompt = """
BOOTING SYSTEM...

DEEPLY ANALYSING TEXTS

START OF TEXTS
<|sequences|>
END OF TEXTS

TASK:
<|task|>

MESSAGE FROM USER: Thank you so much!
"""

task_from_sequence = """
- REPLY ONLY WITH THE COMMON TOPIC(S) BETWEEN THE TEXTS.
- COVER THE TEXTS CONCISELY IN ONE SHORT LINE REGARDLESS OF TEXT NUMBER.
- NEVER WRITE BRACKETS.
- PROCESSING SYSTEM HANDLES SHORT TOPIC(S).
- WRITE GENERALLY FOR THE TEXTS IN THREE OF LESS TOPICS.
- SPECIFIC AS POSSIBLE WHILE ENCOMPASING ALL TEXTS.
- START WITH "TOPIC(S) COMMON IN ALL TEXTS:" AND THEN WRITE THE TOPICS AND THEN END.
- ONLY WRITE NON-TOPIC TEXT IN NEW LINES.
- INCLUDE ALL NECESSARY TOPICS.
- SHORT RESPONSE.
- BE BRIEF BUT HIGHLY INTELLIGENT.
"""

task_from_prev_or_next_latents = """
Thank you for your previous work, you've always done such a great job on this. As usual, your task is to reply with three common topics you find within the texts. Mention keywords and be specific. Earlier texts are more important. If a topic is repeated over texts, write it as one first. Please never write "Interdisciplinary", "disciplines", "complex systems", "across fields", etc. Never write brackets, parentheses, colons, or filler words. Each topic is 3 words or less with no added info, separated by commas. Be sure to cover all relevant topics. Write in sentence capitalisation, but write exceedingly concisely. Thank you! TLDR: Intelligently write 3 short specific topics on one line with no line breaks.
"""



explanations_path = "explanations"
os.makedirs(explanations_path, exist_ok=True)
latents_path = "latents"
latents_filenames_list = [
    {
        "type": "from_sequence",
        "tokens": "latents_avg_mlp_down_tokens_from_sequence.npy",
        "values": "latents_avg_mlp_down_values_from_sequence.pth",
        "save": "latents_avg_mlp_down_explanations_from_sequence.npy"
    },
    {
        "type": "from_sequence",
        "tokens": "latents_avg_sae_tokens_from_sequence.npy",
        "values": "latents_avg_sae_values_from_sequence.pth",
        "save": "latents_avg_sae_explanations_from_sequence.npy"
    },
    {
        "type": "from_prev_latents",
        "tokens": "latents_mlp_down_latents_from_prev_latents.npy",
        "values": "latents_mlp_down_values_from_prev_latents.pth",
        "end_explanations": "latents_avg_mlp_down_explanations_from_sequence.npy",
        "save": "latents_avg_mlp_down_explanations_from_prev_latents.npy"
    },
    {
        "type": "from_next_latents",
        "tokens": "latents_mlp_down_latents_from_next_latents.npy",
        "values": "latents_mlp_down_values_from_next_latents.pth",
        "end_explanations": "latents_avg_mlp_down_explanations_from_sequence.npy",
        "save": "latents_avg_mlp_down_explanations_from_next_latents.npy"
    },
    {
        "type": "from_prev_latents",
        "tokens": "latents_sae_latents_from_prev_latents.npy",
        "values": "latents_sae_values_from_prev_latents.pth",
        "end_explanations": "latents_avg_sae_explanations_from_sequence.npy",
        "save": "latents_avg_sae_explanations_from_prev_latents.npy"
    },
    {
        "type": "from_sequence",
        "tokens": "latents_avg_residuals_tokens_from_sequence.npy",
        "values": "latents_avg_residuals_values_from_sequence.pth",
        "save": "latents_avg_residuals_explanations_from_sequence.npy"
    },
    {
        "type": "from_prev_latents",
        "tokens": "latents_residuals_latents_from_prev_latents.npy",
        "values": "latents_residuals_values_from_prev_latents.pth",
        "end_explanations": "latents_avg_residuals_explanations_from_sequence.npy",
        "save": "latents_avg_residuals_explanations_from_prev_latents.npy"
    },
    {
        "type": "from_next_latents",
        "tokens": "latents_residuals_latents_from_next_latents.npy",
        "values": "latents_residuals_values_from_next_latents.pth",
        "end_explanations": "latents_avg_residuals_explanations_from_sequence.npy",
        "save": "latents_avg_residuals_explanations_from_next_latents.npy"
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
    explanations = np.zeros((latents_tokens.shape[0], latents_tokens.shape[1], explanation_length), dtype=np.uint16)
    if "end_explanations" in latents_filenames:
        end_explanations = np.load(explanations_path + "/" + latents_filenames['end_explanations'])
    
    layers_range = range(latents_tokens.shape[0])
    if latents_filenames["type"] == "from_next_latents":
        layers_range = layers_range[::-1]
        
    for layer_index in layers_range:
        print(f"Layer {layer_index+1} / {latents_tokens.shape[0]}")
        prompts = []
        
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
                
        for latent_index in range(latents_tokens.shape[1]):
            sequences = []
            
            prompt = str(explanation_prompt).strip()
            if latents_filenames["type"] == "from_sequence":
                prompt = prompt.replace("<|task|>", task_from_sequence.strip())
                sequences = [llm.tokenizer.decode(latents_tokens[layer_index][latent_index][i*2]) for i in range(6)]
                sequences = ["Text " + str(i+1) + ": " + sequence.lstrip("<|endoftext|><s>").strip().replace("<|endoftext|>", "<split_in_text>").replace("<s>", "") for i, sequence in enumerate(sequences)]
                prompt = prompt.replace("<|sequences|>", "\n\n".join(sequences))

            elif latents_filenames["type"] == "from_prev_latents":
                prompt = prompt.replace("<|task|>", task_from_prev_or_next_latents.strip())
                prev_latent_indices_frequencies = {}
                for i in range(4):
                    for prev_latent_index in latents_tokens[layer_index][latent_index][i*2]:
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
                sequences = ["- " + prev_latent_explanation for i, prev_latent_explanation in enumerate(prev_latent_explanations)]
                prompt = prompt.replace("<|sequences|>", "\n".join(sequences))

            elif latents_filenames["type"] == "from_next_latents":
                prompt = prompt.replace("<|task|>", task_from_prev_or_next_latents.strip())
                next_latent_indices_frequencies = {}
                for i in range(4):
                    for next_latent_index in latents_tokens[layer_index][latent_index][i*2]:
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
                sequences = ["- " + next_latent_explanation for i, next_latent_explanation in enumerate(next_latent_explanations)]
                prompt = prompt.replace("<|sequences|>", "\n".join(sequences))
                
            prompt = [{ "role": "user", "content": prompt.strip() }]
            prompts.append(prompt)

        start_time = time.time()
        
        response = llm.generate_batch(prompts, max_length = explanation_length)

        explanations_tokens = []
        for i, explanation in enumerate(response):
            explanation = explanation[len("TOPIC(S) COMMON IN ALL TEXTS:"):].strip().split("\n")[0].lstrip("-").lstrip("1.").strip()
            if i < 12:
                print("")
                print("Latent ", i+1)
                print("")
                print("Prompt:", prompts[i][0]["content"])
                print("")
                print("Explanation:", explanation)
                print("")
                print("")
            tokens = llm.tokenizer.encode(explanation)
            tokens = tokens[:explanation_length]
            tokens = tokens + [1] * (explanation_length - len(tokens))
            explanations_tokens.append(tokens)
        
        explanations[layer_index] = np.array(explanations_tokens, dtype=np.uint16)
            
        print(f"Duration: {time.time()-start_time:.2f}s")
        print("")
        print("")
        print("")
        print("")

    np.save(explanations_path  + "/" + latents_filenames["save"], explanations)

    print("Saved " + latents_filenames["save"])
