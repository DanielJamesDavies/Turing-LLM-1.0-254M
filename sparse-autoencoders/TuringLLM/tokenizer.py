import logging
from transformers import AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)

class Tokenizer:
    def __init__(self, model_id = "microsoft/Phi-3-mini-4k-instruct"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True, _fast_init=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, _fast_init=True)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_eos_token(self):
        return self.tokenizer.eos_token_id

    def get_pad_token(self):
        return self.tokenizer.bos_token_id
