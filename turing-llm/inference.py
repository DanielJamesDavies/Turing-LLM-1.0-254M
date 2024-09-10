import random
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

from model import TuringLLM
from tokenizer import Tokenizer



model_path = "model_1722550239_03986.pt"
model_name = "Turing-LLM-1.0-254M"
max_batch_size = 8



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
            print(index, tokens)
            tokens = tokens[index:]
            text = self.tokenizer.decode(tokens)
            results.append([text, tokens])
        
        return results



model = TuringLLMForInference()
prompt = ""
results = model.generate_batch([prompt], max_length=256)
print("")
for result in results:
    print(result[0])
    print("")
    print("-----------------")
    print("")
