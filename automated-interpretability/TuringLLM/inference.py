import time
import random
import numpy as np
import torch
from torch.nn import functional as F

from TuringLLM.model import TuringLLM
from TuringLLM.tokenizer import Tokenizer



model_path = "TuringLLM/model_1722550239_03986.pt"
model_name = "Turing-LLM-1.0-254M"
max_batch_size = 8



random.seed(12)
    


class TuringLLMForInference:

    def __init__(self, model_path=model_path, collect_latents=False, max_length=72):
        self.tokenizer = Tokenizer()
        self.eos_token_id = self.tokenizer.get_eos_token()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.collect_latents = collect_latents
        self.max_length = max_length

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
        self.model.collect_latents = self.collect_latents
        self.model.max_length = self.max_length
        self.model.to(self.device)
        self.model = torch.compile(self.model)
        self.model.eval()
        
        
        
    def clear(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
    
    
    
    def generate(self, x, max_length=64, tokenize=True, topk=12):
        if tokenize is True:
            tokens = self.tokenizer.encode(x)
            tokens = [self.eos_token_id] + tokens
        else:
            tokens = [self.eos_token_id] + x
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
    
    
    
    def generate_batch(self, batch, max_length=64, tokenize=True, decode=True, ignore_end=False, topk=12):
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(12)
        
        batch_tokens = []
        for x in batch:
            if tokenize is True:
                tokens = self.tokenizer.encode(x)
                tokens = [self.eos_token_id] + tokens
            else:
                tokens = x
            tokens = torch.tensor(tokens, dtype=torch.long)
            batch_tokens.append(tokens)
            
        # def get_bins(batch_tokens):
        #     bins = {}
        #     for tokens in batch_tokens:
        #         if len(tokens) not in bins:
        #             bins[len(tokens)] = []
        #         bins[len(tokens)].append(tokens)
        #     bins_array = []
        #     for key in list(sorted(bins.keys())):
        #         bins_array.append(bins[key])
        #     return bins_array
        
        start_time = time.time()
        
        # max_input_tokens_length = max(len(tokens) for tokens in batch_tokens)
        # min_input_tokens_length = min(len(tokens) for tokens in batch_tokens)
        # if min_input_tokens_length < max_input_tokens_length:
        #     while min_input_tokens_length < max_input_tokens_length:
        #         bins = get_bins(batch_tokens)
        #         tokens = torch.stack([tensor.to(self.device) for tensor in bins[0]])
        #         xgen = tokens.to(self.device)
        #         with torch.no_grad():
        #             logits, _ = self.model(xgen)
        #             logits = logits[:, -1, :]
        #             props = F.softmax(logits, dim=-1)
        #             topk_probs, topk_indices = torch.topk(props, topk, dim=-1)
        #             ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
        #             xcol = torch.gather(topk_indices, -1, ix)
        #             xgen = torch.cat((xgen, xcol), dim=1)
                    
        #         new_batch_tokens = []
        #         for tokens in xgen:
        #             new_batch_tokens.append(tokens)
        #         for bin in bins[1:]:
        #             for tokens in bin:
        #                 new_batch_tokens.append(tokens)
        #         batch_tokens = new_batch_tokens.copy()
                    
        #         min_input_tokens_length = min(len(tokens) for tokens in batch_tokens)
        
        tokens = torch.stack([tensor.to(self.device) for tensor in batch_tokens])
        
        xgen = tokens.to(self.device)
        completed = []
        latents = []
        
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, latents_mlp_down, latents_residuals = self.model(xgen)
                if self.model.collect_latents is True and xgen.size(1) >= max_length - 1:
                    latents = [latents_mlp_down, latents_residuals]
                
                logits = logits[:, -1, :]
                props = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(props, topk, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)

                completed = [xgen[i].tolist() for i in range(xcol.size(0))]

        end_time = time.time()
        tokens_generated_count = 0
        
        results = []
        for tokens in completed:
            if self.eos_token_id in tokens and ignore_end is False:
                index = tokens.index(self.eos_token_id) + 1
                tokens = tokens[index:]
            tokens_generated_count += len(tokens)
            if decode is True:
                text = self.tokenizer.decode(tokens)
                results.append([tokens, text])
            else:
                results.append([tokens])

        generation_rate = tokens_generated_count / (end_time - start_time)
        print(f"Generation Rate: {generation_rate:.4f} tok/sec", end="")
        
        if self.model.collect_latents is True:
            return results, latents
        else:
            return results
