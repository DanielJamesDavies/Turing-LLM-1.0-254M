import os
import random
import numpy as np
from transformers import AutoTokenizer



dataset_path = "auto_interp_dataset"
text_path = "text"
num_shuffles = 4



random.seed(12)



model_id = "microsoft/Phi-3-mini-4k-instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True, _fast_init=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, _fast_init=True)
print("Vocab Size:", len(tokenizer))



def get_text_paths(curr_path, text_paths):
    path_contents = os.listdir(curr_path)
    
    for path_content_filename in path_contents:
        if path_content_filename.split(".")[-1] == "txt":
            text_paths.append(curr_path + "/" + path_content_filename)
        elif "." not in path_content_filename:
            text_paths = get_text_paths(curr_path + "/" + path_content_filename, text_paths)
            
    return text_paths



def get_split_txt(txt):
    split_txt = []
    
    txt = txt.replace("<|endofaugtext|>", "<|endoftext|>")
    txt = txt.replace("<|endoftext|>", "\n")
    for paragraph in txt.split("\n"):
        if len(paragraph.strip()) != 0:
            for text_index, text in enumerate(paragraph.split(". ")):
                if len(text.strip().rstrip(".")) != 0:
                    if text_index == 0:
                        split_txt.append("<|endoftext|><s>" + text.strip().rstrip(".") + ".")
                    else:
                        split_txt.append(text.strip().rstrip(".") + ".")
                
    return split_txt
    
    

def get_tokens_from_split_txt(split_txt):
    prompt_token_count = 64
    
    prompt_tokens = []
    all_tokens = []
    tokens_remaining = 0
    
    for text_index, text in enumerate(split_txt):
        tokens = np.array(tokenizer.encode(text)).astype(np.uint16)
        all_tokens.append(tokens[1:])
        tokens_remaining += len(tokens[1:])
        
    for i, tokens in enumerate(all_tokens):
        tokens_remaining -= len(tokens)
        
        if len(tokens) >= prompt_token_count:
            prompt_tokens.append(tokens[:prompt_token_count])
            
        else:
            
            new_tokens = tokens
            curr_i = int(i)
            
            if i != len(all_tokens) - 1 and tokens_remaining >= prompt_token_count: # All but last
                while len(new_tokens) < prompt_token_count and curr_i < len(all_tokens) - 2:
                    curr_i += 1
                    new_tokens = np.concatenate((new_tokens, all_tokens[curr_i]))
                prompt_tokens.append(new_tokens[:prompt_token_count])
            else: # Last
                while len(new_tokens) < prompt_token_count and curr_i > 0:
                    curr_i += -1
                    new_tokens = np.concatenate((all_tokens[curr_i], new_tokens))
                prompt_tokens.append(new_tokens[-prompt_token_count:])
    
    def filter_prompt_tokens(tokens):
        if len(tokens) != prompt_token_count:
            return False
        return True
    
    prompt_tokens = list(filter(filter_prompt_tokens, prompt_tokens))
    
    # for prompt in prompt_tokens[:5]:
    #     print(tokenizer.decode(prompt))
    #     print("")
    
    return prompt_tokens



def create_shards():
    print("")
    print("Creating Shards")
    
    text_paths = get_text_paths(text_path, [])
    random.shuffle(text_paths)
    
    sequences = []
    
    for i, txt_path in enumerate(text_paths):
        if i == 0 or i % 250 == 250-1 or i == len(text_paths)-1:
            print(f"Text {i+1} / {len(text_paths)} ({(i+1)/len(text_paths)*100:.2f}%)")
        txt = open(txt_path, 'r', encoding='utf-8').read()
        split_txt = get_split_txt(txt)
        tokens = get_tokens_from_split_txt(split_txt)
        sequences.extend(tokens)
        
    for _ in range(num_shuffles):
        random.shuffle(sequences)
        
    print("Num Sequences: ", len(sequences))
    
    shard_count = 4096 # 8192
    shards = np.array_split(sequences, shard_count)
    for shard_index, shard in enumerate(shards):
        separator = np.array([-1])
        shard = np.concatenate([np.concatenate([sequence, separator]) for sequence in shard])
        np.save(f'{dataset_path}/shard_{shard_index}.npy', shard)



def run():
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)

    # Create Shards
    create_shards()
    
    print("Complete.")
    


run()
