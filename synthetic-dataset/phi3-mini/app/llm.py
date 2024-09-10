import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import atexit
import gc

class LLM:
    _instance = None

    class __LLM:
    
        def __init__(self, model_id="microsoft/Phi-3-mini-4k-instruct"):
            atexit.register(self.cleanup)
            self.model_id = model_id
            self.setup_model()
            
        def setup_model(self):            
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

        def generate(self, messages, max_length=8000, skip_special_tokens=True):
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            input_ids = input_ids.to(self.device)
            
            max_new_tokens = min(max_length, self.tokenizer.model_max_length - input_ids.shape[1])
            
            start_time = time.time()
            
            generation_args = {
                "return_tensors": True,
                "max_new_tokens": max_new_tokens,
            }

            with torch.no_grad():
                output = self.pipe(messages, **generation_args)
            
            def get_last_assistant_token_index(tokens):
                for i in range(len(tokens)-1,-1,-1):
                    if tokens[i] == 32001:
                        return i

            last_assistant_token_index = get_last_assistant_token_index(output[0]['generated_token_ids'])

            generated_tokens = output[0]['generated_token_ids'][last_assistant_token_index+1:]
            decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=skip_special_tokens)

            end_time = time.time()
            seconds = end_time - start_time
            tokens_generated_count = len(output[0]['generated_token_ids'][last_assistant_token_index+1:])
            generation_rate = tokens_generated_count / seconds

            print(f"Single Generation - Generation Rate: {generation_rate:.4f} tok/sec")
            
            del output

            return decoded_output, generated_tokens

        def generate_batch(self, batch, max_length=8000, skip_special_tokens=True, reset_on_low_tok_sec=False, batch_size=8):
            new_batch = []
            max_input_token_length = 0
            for i in range(len(batch)):
                new_batch.append(self.tokenizer.apply_chat_template(batch[i], tokenize=False, add_generation_prompt=True))
                tokens = self.tokenizer.apply_chat_template(batch[i], tokenize=True, add_generation_prompt=True, return_tensors="pt")
                max_input_token_length = max(max_input_token_length, tokens.shape[1])
            
            def get_messages():
                for messages in new_batch:
                    yield messages
            
            generation_args = {
                "batch_size": min(8, len(batch), batch_size),
                "return_tensors": True,
                "max_new_tokens": min(max_length, self.tokenizer.model_max_length - max_input_token_length),
                "do_sample": True,
                "top_k": 20
            }
            
            start_time = time.time()
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
            tokens_generated_count = 0
            for output in outputs:
                last_assistant_token_index = get_last_assistant_token_index(output)
                generated_tokens = output[last_assistant_token_index+1:]
                decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=skip_special_tokens)
                generated_texts.append(decoded_output)
                tokens_generated_count += len(generated_tokens)

            end_time = time.time()
            seconds = end_time - start_time
            generation_rate = tokens_generated_count / seconds

            print(f"Batch Generation - Generation Rate: {generation_rate:.4f} tok/sec")
            
            if reset_on_low_tok_sec and generation_rate < 75:
                self.reset()

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

    def __new__(cls):
        if not cls._instance:
            cls._instance = cls.__LLM()
        return cls._instance
