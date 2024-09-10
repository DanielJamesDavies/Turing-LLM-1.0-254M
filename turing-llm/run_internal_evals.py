import torch
from torch.nn import functional as F
import random
import json
from dataclasses import dataclass
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc

from model import TuringLLM
from tokenizer import Tokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)



model_path = "model_1722550239_03986.pt"
model_name = "Turing-LLM-1.0-254M"



@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5
    


class TuringLLMForInference:

    def __init__(self, model_path=model_path):
        self.tokenizer = Tokenizer()
        self.eos_token_id = self.tokenizer.get_eos_token()
        self.bos_token_id = self.tokenizer.get_bos_token()
        self.pad_token_id = self.tokenizer.get_pad_token()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.manual_seed(12)
        if self.device == 'cuda':
            torch.cuda.manual_seed(12)
        torch.set_float32_matmul_precision('high')
        
        self.load_model(model_path)



    def load_model(self, checkpoint_path):
        if checkpoint_path is False:
            return False, False, False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        random.setstate(checkpoint['rng_states']['python'])
        np.random.set_state(checkpoint['rng_states']['numpy'])
        torch.set_rng_state(torch.ByteTensor(checkpoint['rng_states']['torch']))
        if torch.cuda.is_available() and checkpoint['rng_states']['cuda'] is not None:
            torch.cuda.set_rng_state_all([torch.ByteTensor(s) for s in checkpoint['rng_states']['cuda']])
        
        self.model = TuringLLM(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model = torch.compile(self.model)
        self.model.eval()
    
    
    
    def generate(self, x, max_length=64, topk=12):
        tokens = self.tokenizer.encode(x)
        tokens = [self.eos_token_id] + tokens
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(1, 1)
        xgen = tokens.to(self.device)
        
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(12)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, _ = self.model(xgen)
                logits = logits[:, -1, :]
                props = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(props, topk, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)

        tokens = xgen[0, :max_length].tolist()
        text = self.tokenizer.decode(tokens[1:])
        return text, tokens
    
    
    
    def generate_batch(self, batch, max_length=64, topk=12):
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(12)
        
        batch_tokens = []
        for x in batch:
            tokens = self.tokenizer.encode(x)
            tokens = [self.eos_token_id, self.bos_token_id] + tokens
            tokens = torch.tensor(tokens, dtype=torch.long)
            batch_tokens.append(tokens)
            
        def get_bins(batch_tokens):
            bins = {}
            for tokens in batch_tokens:
                if len(tokens) not in bins:
                    bins[len(tokens)] = []
                bins[len(tokens)].append(tokens)
            bins_array = []
            for key in list(sorted(bins.keys())):
                bins_array.append(bins[key])
            return bins_array
        
        max_input_tokens_length = max(len(tokens) for tokens in batch_tokens)
        min_input_tokens_length = min(len(tokens) for tokens in batch_tokens)
        while min_input_tokens_length < max_input_tokens_length:
            bins = get_bins(batch_tokens)
            tokens = torch.stack([tensor.to(self.device) for tensor in bins[0]])
            xgen = tokens.to(self.device)
            with torch.no_grad():
                logits, _ = self.model(xgen)
                logits = logits[:, -1, :]
                props = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(props, topk, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
                
            new_batch_tokens = []
            for tokens in xgen:
                new_batch_tokens.append(tokens)
            for bin in bins[1:]:
                for tokens in bin:
                    new_batch_tokens.append(tokens)
            batch_tokens = new_batch_tokens.copy()
                
            min_input_tokens_length = min(len(tokens) for tokens in batch_tokens)
        
        tokens = torch.stack([tensor.to(self.device) for tensor in batch_tokens])
        
        xgen = tokens.to(self.device)
        
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, _ = self.model(xgen)
                logits = logits[:, -1, :]
                props = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(props, topk, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)

        results = []
        for i in range(len(batch)):
            tokens = xgen[i, :max_length].tolist()
            index = tokens.index(self.bos_token_id) + 1
            tokens = tokens[index:]
            text = self.tokenizer.decode(tokens)
            results.append([text, tokens])
        
        return results



class Phi3_Mini:
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

    def generate_batch(self, batch, initial_assistant_message="", max_new_tokens=512, batch_size=8, skip_special_tokens=True, reset_on_low_tok_sec=False):
        new_batch = []
        max_input_token_length = 0
        for i in range(len(batch)):
            new_batch.append(self.tokenizer.apply_chat_template(batch[i], tokenize=False, add_generation_prompt=True) + initial_assistant_message)
            tokens = self.tokenizer.apply_chat_template(batch[i], tokenize=True, add_generation_prompt=True, return_tensors="pt")
            max_input_token_length = max(max_input_token_length, tokens.shape[1])
        
        def get_messages():
            for messages in new_batch:
                yield messages
        
        generation_args = {
            "batch_size": batch_size,
            "return_tensors": True,
            "max_new_tokens": max(1, max_new_tokens),
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
        
        generated = []
        for output in outputs:
            last_assistant_token_index = get_last_assistant_token_index(output)
            generated_tokens = output[last_assistant_token_index+1:]
            decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=skip_special_tokens)
            generated.append([decoded_output, generated_tokens])

        return generated

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



def run():
    turing_llm = TuringLLMForInference()
    phi3_mini = Phi3_Mini()
    
    english_quality_grades = {}
    topic_adherence_grades = {}
    
    subjects = "Physics\nMathematics\nComputer Science\nChemistry\nBiology\nPhilosophy\nEnglish\nCreative Writing\nLanguages\nArt\nMusic\nEngineering\nEconomics\nBusiness\nGeography\nHistory\nCulinary Arts\nCommunication"
    
    for subject in subjects.split("\n"):
        subject_key = subject.lower().replace(" ", "_")
        english_quality_grades[subject_key] = []
        topic_adherence_grades[subject_key] = []
        print("")
        print("")
        print("")
        print("Subject: ", subject)
        print("")
        print("")
        print("")
        generated = phi3_mini.generate_batch([[{ "role": "user", "content": f"Concisely, please write an ordered line-separated list of 24 different topics in the subject of {subject}." }]])
        topics = [topic.strip()[2:].lstrip(".").lstrip(")").strip() for topic in generated[0][0].split("\n")]
        print(f"Topics ({len(topics)}):", topics)
        print("")
        print("")
    
        for topic in topics:
            print("")
            print("Topic:", topic)
            
            generated = phi3_mini.generate_batch([[{ "role": "user", "content": f"Please write one sentence about {topic}." }]])
            phi_3_mini_initial_text = generated[0][0]
            print("")
            print(phi_3_mini_initial_text)
            
            turing_max_length = len(generated[0][1]) + 64
            turing_generated_texts = turing_llm.generate_batch([phi_3_mini_initial_text], max_length=turing_max_length)
            
            for turing_generated_text in turing_generated_texts:
                turing_generated_text = turing_generated_text[0][len(phi_3_mini_initial_text):].strip().rstrip(",")
                if turing_generated_text[-1] not in [".", "!", "?"]:
                    turing_generated_text += "..."
                print("")
                print(turing_generated_text)
                
                english_quality_eval_prompt = "Please give a grade (A-F) for the quality of English in this extract from a text: "
                initial_assistant_message = "Grade:"
                generated_texts = phi3_mini.generate_batch([[{ "role": "user", "content": english_quality_eval_prompt + turing_generated_text }]], initial_assistant_message=initial_assistant_message, max_new_tokens=2)
                english_quality_eval_response = generated_texts[0][0].strip().split("\n")[0][len(initial_assistant_message):].strip()[0]
                print("")
                print("English Quality Grade: ", english_quality_eval_response)
                english_quality_grades[subject_key].append(english_quality_eval_response.lower())
                
                topic_adherence_eval_prompt = "Please give a grade (A-F) for how relevant the text extract is to the topic.\nTopic: <|topic|>\nText Extract: "
                topic_adherence_eval_prompt = topic_adherence_eval_prompt.replace("<|topic|>", topic)
                initial_assistant_message = "Grade:"
                generated_texts = phi3_mini.generate_batch([[{ "role": "user", "content": topic_adherence_eval_prompt + turing_generated_text }]], initial_assistant_message=initial_assistant_message, max_new_tokens=2)
                topic_adherence_eval_response = generated_texts[0][0].strip().split("\n")[0][len(initial_assistant_message):].strip()[0]
                print("")
                print("Topic Adherence Grade: ", topic_adherence_eval_response)
                topic_adherence_grades[subject_key].append(topic_adherence_eval_response.lower())
                
                print("")
                print("")
                
    with open('internal_eval_results.json', 'w', encoding='utf-8') as f:
        json.dump({ "english_quality_grades": english_quality_grades, "topic_adherence_grades": topic_adherence_grades }, f)

run()
