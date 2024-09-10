import time
import os
from dataclasses import dataclass
import numpy as np
import torch
import gc
import shutil

from TuringLLM.inference_2 import TuringLLMForInference
from SAE_TopK import SAE, SAETrainer



dataset_path = "../synthetic-dataset/auto_interp_dataset"
sae_path = "sae"



@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5


d_sae = 10240 # 48000
d_model = TuringLLMConfig.n_embd



device = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_shard_count():
    dataset_contents = os.listdir(dataset_path)
    shard_count = 0
    for dataset_content in dataset_contents:
        if ".npy" in dataset_content:
            shard_count += 1
    return shard_count


def collect_activations(shard_index, shard_count):
    
    shard = np.load(f"{dataset_path}/shard_{shard_index}.npy")
    shard = np.split(shard, np.where(shard == -1)[0])
    shard = [sequence[sequence != -1] for sequence in shard if sequence.size > 1]
    
    torch.cuda.empty_cache()
    gc.collect()
    
    turing = TuringLLMForInference(collect_activations=True)
    turing_batch_size = 512
    
    
    activations = [False for _ in range(TuringLLMConfig.n_layer)]
    
    for i in range(0, len(shard), turing_batch_size):
        start_time = time.time()
        
        total_batches_length = (len(shard) + turing_batch_size - 1) // turing_batch_size
        print(f"Turing Inference  |  Shard {shard_index + 1} / {shard_count}  |  Batch {str(i // turing_batch_size + 1).zfill(len(str(total_batches_length)))} / {total_batches_length}  |  ", end="")
        
        batch = shard[i:i + turing_batch_size]
        results, acts = turing.generate_batch([sequence.tolist() for sequence in batch], max_length=65, tokenize=False, decode=False, ignore_end=True)
        
        for layer_index, layer_acts in enumerate(acts[1]):
            if activations[layer_index] is False:
                activations[layer_index] = layer_acts.view(-1, 1024)
            else:
                activations[layer_index] = torch.cat((activations[layer_index], layer_acts.view(-1, 1024)), dim=0)
                
        print(f"  |  Duration: {time.time()-start_time:.2f}s")
    
    turing.clear()
    del turing
    
    print(f"Activations Collected Per Layer: {activations[0].shape[0]:,}")
    
    return activations



def train_sparse_autoencoders(shard_index, activations, step_offsets):
    for layer_index in range(TuringLLMConfig.n_layer):
        print("")
        
        # Load SAE
        k = 128 + (layer_index * 16)
        sae = SAE(d_model, d_sae, k)
        sae = torch.compile(sae)
        print(f"SAE Params: {sae.get_num_params():,}")
        print(f"SAE K:      {k}")
        if shard_index != 0:
            sae.load(f"{sae_path}/sae_layer_{layer_index}.pth")
            
        # Train SAE
        sae_trainer = SAETrainer(sae, layer_index, TuringLLMConfig.n_layer - 1, sae_path)
        
        data = [t.squeeze(0) for t in torch.split(activations[layer_index], 1, dim=0)]
        final_step = sae_trainer.train(data, step_offsets[layer_index])
        step_offsets[layer_index] = final_step + 1
        
        # Save SAE
        sae.save(f"{sae_path}/sae_layer_{layer_index}.pth")
        
        # Del SAE
        del sae
        torch.cuda.empty_cache()
        gc.collect()
    
    return step_offsets
    


def run():
    current_path = os.path.dirname(__file__)
    os.makedirs(sae_path, exist_ok=True)
    
    shard_start = 878
    step_offset = 7016
    
    shard_count = get_shard_count()
    # shard_count = min(shard_count, 724)
    step_offsets = [step_offset for _ in range(TuringLLMConfig.n_layer)]
    
    for shard_index in range(shard_count)[shard_start-1:]:
        start_time = time.time()
        activations = collect_activations(shard_index, shard_count)
        step_offsets = train_sparse_autoencoders(shard_index, activations, step_offsets)
        del activations
        print("")
        print(f"Shard Duration: {time.time()-start_time:.2f}s")
        print("")
        print("")
        
        if shard_index % 12 == 0:
            print("Backing Up...")
            print("")
            print("")
            os.makedirs("sae-backup", exist_ok=True)
            shutil.copytree(current_path + "/" + sae_path, current_path + "/sae-backup-temp")
            with open(current_path + "/sae-backup-temp/save.txt", "w") as f:
                step_offsets_string = " ".join([str(step_offsets) for step_offsets in step_offsets])
                f.write(f"shard_index {str(shard_index)}\nstep_offsets {str(step_offsets_string)}")
            shutil.rmtree(current_path + "/sae-backup")
            os.rename("sae-backup-temp", "sae-backup")

run()



