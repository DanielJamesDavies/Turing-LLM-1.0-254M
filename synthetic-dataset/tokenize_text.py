import os
import numpy as np
from transformers import AutoTokenizer



tokens_path = "tokens"
text_path = "text"
shard_size = 100000000



model_id = "microsoft/Phi-3-mini-4k-instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True, _fast_init=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, _fast_init=True)
print("Vocab Size:", len(tokenizer))



def tokenize(text):
    text = text.replace("<|endofaugtext|>", "<|endoftext|>")
    tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
    tokens_np = np.array(tokens)
    return tokens_np.astype(np.uint16)



def save_shard(shard_index, shard_tokens_np, shard_token_count, save_path):
    print(f"Saving Shard: {save_path}/shard_{shard_index}")
    np.save(f"{save_path}/shard_{shard_index}", shard_tokens_np[:shard_token_count])
    shard_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    shard_token_count = 0
    return shard_index + 1, shard_tokens_np, shard_token_count



def process_path(path, save_path, shard_index, shard_tokens_np, shard_token_count):
    print("Processing Path: " + str(path))
    path_contents = os.listdir(path)
    
    def filter_txt_files(name):
        if name[-4:] == ".txt":
            return True
        return False
    
    def filter_folders(name):
        if "." in name:
            return False
        return True
    
    path_txt_files = list(filter(filter_txt_files, path_contents))
    path_folders = list(filter(filter_folders, path_contents))
    
    # For each file
    for path_txt_file in path_txt_files:
        text = open(path + "/" + path_txt_file, 'r', encoding='utf-8').read()
        tokens = tokenize(text)
        
        # Save Shard
        if shard_token_count + len(tokens) > shard_size:
            shard_index, shard_tokens_np, shard_token_count = save_shard(shard_index, shard_tokens_np, shard_token_count, save_path)
        
        # Add Tokens to Shard
        shard_tokens_np[shard_token_count:shard_token_count+len(tokens)] = tokens
        shard_token_count += len(tokens)
    
    # For each child
    for path_folder in path_folders:
        shard_index, shard_tokens_np, shard_token_count = process_path(path + "/" + path_folder, save_path, shard_index, shard_tokens_np, shard_token_count)
    
    return shard_index, shard_tokens_np, shard_token_count



def run():
    os.makedirs(tokens_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)

    shard_index = 0
    shard_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    shard_token_count = 0
    
    os.makedirs(tokens_path + "/phi-3-mini", exist_ok=True)
    shard_index, shard_tokens_np, shard_token_count = process_path(text_path + "/phi-3-mini", tokens_path + "/phi-3-mini", shard_index, shard_tokens_np, shard_token_count)
    shard_index, shard_tokens_np, shard_token_count = save_shard(shard_index, shard_tokens_np, shard_token_count, tokens_path + "/phi-3-mini")
    
    os.makedirs(tokens_path + "/data-augmentation", exist_ok=True)
    shard_index, shard_tokens_np, shard_token_count = process_path(text_path + "/data-augmentation", tokens_path + "/data-augmentation", shard_index, shard_tokens_np, shard_token_count)
    shard_index, shard_tokens_np, shard_token_count = save_shard(shard_index, shard_tokens_np, shard_token_count, tokens_path + "/data-augmentation")
    
    print("Complete.")
    


run()
