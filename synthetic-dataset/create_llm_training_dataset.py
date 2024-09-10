import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer
import shutil



dataset_path = "llm_training_dataset"
tokens_path = "tokens"
num_shuffles = 64



random.seed(12)
        
        
        
model_id = "microsoft/Phi-3-mini-4k-instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True, _fast_init=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, _fast_init=True)
eos_token_id = tokenizer.eos_token_id
del tokenizer
torch.cuda.empty_cache()
print("eos_token_id:", eos_token_id)



def shuffle_shards(input_shard_path, output_shard_path, num_shuffles):
    print(f"Creating Shards for {output_shard_path}")
    num_shards = 0
    
    os.makedirs(input_shard_path, exist_ok=True)
    os.makedirs(output_shard_path, exist_ok=True)
    
    # Create output_shard_path shards
    for filename in os.listdir(input_shard_path):
        if filename.split(".")[-1] == "npy" and "val" not in filename and filename != "phi-3-mini":
            shutil.copy(input_shard_path + "/" + filename, output_shard_path + f"/train_{num_shards}.npy")
            num_shards += 1
        elif "." not in filename:
            if "val" in filename or filename == "phi-3-mini":
                continue
            folder = "/" + filename
            for filename in os.listdir(input_shard_path + folder):
                if filename.split(".")[-1] == "npy" and "val" not in filename:
                    shutil.copy(input_shard_path + folder + "/" + filename, output_shard_path + f"/train_{num_shards}.npy")
                    num_shards += 1
    
    print("  Num Shards: ", num_shards)
    
    # Get Shards
    new_shards = []
    for shard_index in range(num_shards):
        shard = np.load(output_shard_path + f"/train_{shard_index}.npy")
        
        split_indices = np.where(shard == eos_token_id)[0]
        sub_arrays = np.split(shard, split_indices[:-1])
        if shard[-1] != eos_token_id:
            sub_arrays[-1] = shard[split_indices[-1]-1:]
        new_shard = [sub_array for sub_array in sub_arrays if sub_array.size > 0]
            
        new_shards.append(new_shard)
        
    # Shuffle Sequences
    old_shards = new_shards.copy()
    for shuffle_index in range(num_shuffles):
        if shuffle_index == 0 or (shuffle_index+1) % 4 == 0:
            print(f"  Shuffle {str(shuffle_index+1).zfill(len(str(num_shuffles)))} / {num_shuffles}")
        # Shuffle Sequences in Shard
        for shard_index in range(num_shards):
            random.shuffle(new_shards[shard_index])
        
        # Shuffle Sequences between Shards
        old_shards = new_shards.copy()
        new_shards = [[] for _ in range(num_shards)]
        
        for shard_index in range(num_shards):
            old_shard_np = np.array(old_shards[shard_index], dtype=object)
            split_old_shard_np = np.array_split(old_shard_np, num_shards)
            split_old_shard = [list(shard) for shard in split_old_shard_np]
            for i, shard in enumerate(split_old_shard):
                new_shards[i].extend(shard)
                
    # Save Shards
    for shard_index in range(num_shards):
        new_shard = new_shards[shard_index]
        new_shard_np = np.concatenate(new_shard).astype(np.uint16)
        np.save(output_shard_path + f"/train_{shard_index}.npy", new_shard_np)
        
        

def create_phi_3_mini_dataset_split(input_shard_path, output_shard_path, num_shuffles):
    print("Creating Phi-3 Mini Split Dataset")
    os.makedirs(input_shard_path, exist_ok=True)
    os.makedirs(output_shard_path, exist_ok=True)
    
    num_shards = 0
    
    # Create output_shard_path shards
    for filename in os.listdir(input_shard_path):
        if filename.split(".")[-1] == "npy" and "val" not in filename:
            shutil.copy(input_shard_path + "/" + filename, output_shard_path + f"/train_{num_shards}.npy")
            num_shards += 1
    
    # Get Shards
    new_shards = []
    for shard_index in range(num_shards):
        shard = np.load(output_shard_path + f"/train_{shard_index}.npy")
        
        split_indices = np.where(shard == eos_token_id)[0]
        sub_arrays = np.split(shard, split_indices[:-1])
        if shard[-1] != eos_token_id:
            sub_arrays[-1] = shard[split_indices[-1]-1:]
        new_shard = [sub_array for sub_array in sub_arrays if sub_array.size > 0]
            
        new_shards.append(new_shard)
        
    # Shuffle Sequences
    old_shards = new_shards.copy()
    for shuffle_index in range(num_shuffles):
        if shuffle_index == 0 or (shuffle_index+1) % 4 == 0:
            print(f"  Shuffle {str(shuffle_index+1).zfill(len(str(num_shuffles)))} / {num_shuffles}")
        # Shuffle Sequences in Shard
        for shard_index in range(num_shards):
            random.shuffle(new_shards[shard_index])
        
        # Shuffle Sequences between Shards
        old_shards = new_shards.copy()
        new_shards = [[] for _ in range(num_shards)]
        
        for shard_index in range(num_shards):
            old_shard_np = np.array(old_shards[shard_index], dtype=object)
            split_old_shard_np = np.array_split(old_shard_np, num_shards)
            split_old_shard = [list(shard) for shard in split_old_shard_np]
            for i, shard in enumerate(split_old_shard):
                new_shards[i].extend(shard)
    
    val_shard = []
    for shard_index in range(num_shards):
        old_shard = new_shards[shard_index]
        split_index = int(len(old_shard) * 0.025)
        split_index = max(1, split_index)
        new_shard = old_shard[split_index:]
        new_shard_np = np.concatenate(new_shard).astype(np.uint16)
        np.save(output_shard_path + f"/train_{shard_index}.npy", new_shard_np)
        
        new_val_shard_addition = old_shard[:split_index]
        val_shard.extend(new_val_shard_addition)
    
    val_shard_np = np.concatenate(val_shard).astype(np.uint16)
    np.save(output_shard_path + "/val_0.npy", val_shard_np)


def run():
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(tokens_path, exist_ok=True)
    
    # Create Phi-3 Mini Dataset Split
    create_phi_3_mini_dataset_split(tokens_path + "/phi-3-mini", tokens_path + "/phi-3-mini-split", num_shuffles)
    
    # Create Val Dataset
    os.makedirs(dataset_path + "/val", exist_ok=True)
    shutil.copy(tokens_path + "/phi-3-mini-split/val_0.npy", dataset_path + "/val/val_0.npy")
    
    # First Part Epoch 1
    shuffle_shards(tokens_path, dataset_path + "/p1-e1", num_shuffles)
    
    # # First Part Epoch 2
    shuffle_shards(tokens_path, dataset_path + "/p1-e2", num_shuffles)
    
    # # Second Part Epoch 1
    shuffle_shards(tokens_path + "/phi-3-mini-split", dataset_path + "/p2-e1", num_shuffles)


run()
