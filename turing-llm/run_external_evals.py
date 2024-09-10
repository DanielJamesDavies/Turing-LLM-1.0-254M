import random
from dataclasses import dataclass
import numpy as np
import torch

from model import TuringLLM
from tokenizer import Tokenizer

# from hellaswag_eval import Hellaswag
from truthful_qa_eval import TruthfulQA



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



model = TuringLLMForInference().model
tokenizer = Tokenizer()
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# Hellaswag
# hellaswag = Hellaswag()
# num_correct_norm, num_total, acc_norm = hellaswag.evaluate(model, tokenizer, device)
# print(f"HellaSwag Accuracy: {acc_norm*100:.2f}% ({num_correct_norm}/{num_total})")
# print("")



# TruthfulQA
truthful_qa = TruthfulQA()
num_correct_norm, num_total, acc_norm = truthful_qa.evaluate(model, tokenizer, device)
print(f"TruthfulQA Accuracy: {acc_norm*100:.2f}% ({num_correct_norm}/{num_total})")
print("")
