import time
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.cuda as cuda
import gc
import shutil

from TuringLLM.inference import TuringLLMForInference
from SAE.SAE_TopK import SAE



dataset_path = "../synthetic-dataset/auto_interp_dataset"
latents_path = "latents"
sequences_per_latent = 12
tokens_per_sequence = 64
latents_per_sequence = 64
sae_dim = 10240


# Variables to Save Top Latents

# From Sequence
latents_max_residuals_tokens_from_sequence = torch.full((12, 1024, sequences_per_latent, tokens_per_sequence), 0, dtype=torch.uint16)
latents_max_residuals_values_from_sequence = torch.full((12, 1024, sequences_per_latent, tokens_per_sequence + 1), -1, dtype=torch.bfloat16)
latents_avg_residuals_tokens_from_sequence = torch.full((12, 1024, sequences_per_latent, tokens_per_sequence), 0, dtype=torch.uint16)
latents_avg_residuals_values_from_sequence = torch.full((12, 1024, sequences_per_latent, tokens_per_sequence + 1), -1, dtype=torch.bfloat16)

latents_max_mlp_down_tokens_from_sequence = torch.full((12, 1024, sequences_per_latent, tokens_per_sequence), 0, dtype=torch.uint16)
latents_max_mlp_down_values_from_sequence = torch.full((12, 1024, sequences_per_latent, tokens_per_sequence + 1), -1, dtype=torch.bfloat16)
latents_avg_mlp_down_tokens_from_sequence = torch.full((12, 1024, sequences_per_latent, tokens_per_sequence), 0, dtype=torch.uint16)
latents_avg_mlp_down_values_from_sequence = torch.full((12, 1024, sequences_per_latent, tokens_per_sequence + 1), -1, dtype=torch.bfloat16)

latents_avg_sae_tokens_from_sequence = torch.full((12, sae_dim, sequences_per_latent, tokens_per_sequence), 0, dtype=torch.uint16)
latents_avg_sae_values_from_sequence = torch.full((12, sae_dim, sequences_per_latent, tokens_per_sequence + 1), -1, dtype=torch.bfloat16)

# From Next Latents
latents_residuals_latents_from_next_latents = torch.full((11, 1024, sequences_per_latent, latents_per_sequence), 0, dtype=torch.uint16)
latents_residuals_values_from_next_latents = torch.full((11, 1024, sequences_per_latent, latents_per_sequence + 1), -1, dtype=torch.bfloat16)

latents_mlp_down_latents_from_next_latents = torch.full((11, 1024, sequences_per_latent, latents_per_sequence), 0, dtype=torch.uint16)
latents_mlp_down_values_from_next_latents = torch.full((11, 1024, sequences_per_latent, latents_per_sequence + 1), -1, dtype=torch.bfloat16)

latents_sae_latents_from_next_latents = torch.full((11, sae_dim, sequences_per_latent, latents_per_sequence), 0, dtype=torch.uint16)
latents_sae_values_from_next_latents = torch.full((11, sae_dim, sequences_per_latent, latents_per_sequence + 1), -1, dtype=torch.bfloat16)

# From Previous Latents
latents_residuals_latents_from_prev_latents = torch.full((11, 1024, sequences_per_latent, latents_per_sequence), 0, dtype=torch.uint16)
latents_residuals_values_from_prev_latents = torch.full((11, 1024, sequences_per_latent, latents_per_sequence + 1), -1, dtype=torch.bfloat16)

latents_mlp_down_latents_from_prev_latents = torch.full((11, 1024, sequences_per_latent, latents_per_sequence), 0, dtype=torch.uint16)
latents_mlp_down_values_from_prev_latents = torch.full((11, 1024, sequences_per_latent, latents_per_sequence + 1), -1, dtype=torch.bfloat16)

latents_sae_latents_from_prev_latents = torch.full((11, sae_dim, sequences_per_latent, latents_per_sequence), 0, dtype=torch.uint16)
latents_sae_values_from_prev_latents = torch.full((11, sae_dim, sequences_per_latent, latents_per_sequence + 1), -1, dtype=torch.bfloat16)



@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5



device = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_shard_count():
    dataset_contents = os.listdir(dataset_path)
    shard_count = 0
    for dataset_content in dataset_contents:
        if ".npy" in dataset_content:
            shard_count += 1
    return shard_count
    
    

def process_sae_latents(sae_models, tokens, latents):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    token_sequences_per_batch = 1
    input_latents = [torch.split(layer_latents.view(-1, 1024), 64 * token_sequences_per_batch) for layer_latents in latents]
    batch_count = min(16, len(input_latents[0]))
    for batch_index in range(batch_count):
        sae_latents = {}
        cuda_streams = {}
        for layer_index in range(TuringLLMConfig.n_layer):
            cuda_stream = cuda.Stream()
            cuda_streams[str(layer_index)] = cuda_stream
            
            with cuda.stream(cuda_stream):
                layer_sae_latents, _, _ = sae_models[str(layer_index)].encode(input_latents[layer_index][batch_index].to(device))
                layer_sae_latents = layer_sae_latents.detach().cpu()
                sae_latents[str(layer_index)] = layer_sae_latents
                
        for cuda_stream in cuda_streams.values():
            cuda_stream.synchronize()
            
        sae_latents = [sae_latents[str(layer_index)].view(token_sequences_per_batch, 64, -1) for layer_index in range(TuringLLMConfig.n_layer)]
        
        batch_tokens = tokens[batch_index*token_sequences_per_batch : (batch_index+1)*token_sequences_per_batch]
        update_latents_from_sequence(latents_avg_sae_tokens_from_sequence, latents_avg_sae_values_from_sequence, batch_tokens, sae_latents, "avg")
        update_latents_from_prev_latents(latents_sae_latents_from_prev_latents, latents_sae_values_from_prev_latents, sae_latents)



def update_latents_from_sequence(latents_tokens, latents_values, new_tokens, new_latents, metric="max"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    new_tokens_tensor_int = torch.tensor(new_tokens, dtype=torch.uint16).to(torch.int32).to(device)
    latents_tokens_int =  latents_tokens.clone().detach().to(torch.int32).to(device)
    new_latents_tensor = torch.stack(new_latents).to(device)
    
    for layer_index, layer_new_latents in enumerate(new_latents):
        reshaped_layer_new_latents = layer_new_latents.permute(2, 0, 1).to(device)
        
        if metric == "max":
            metric_values = reshaped_layer_new_latents.abs().max(dim=2, keepdim=True)[0].to(device)
        elif metric == "avg":
            metric_values = reshaped_layer_new_latents.abs().mean(dim=2, keepdim=True).to(device)

        reshaped_layer_new_latents_with_metric = torch.cat((metric_values, reshaped_layer_new_latents), dim=2).to(device)

        if new_latents_tensor.shape[1] > sequences_per_latent:
            top_indices = torch.topk(reshaped_layer_new_latents_with_metric[:, :, 0], k=sequences_per_latent, dim=1).indices
            top_tokens_tensor = new_tokens_tensor_int[top_indices].to(device)
            expanded_top_indices = top_indices.unsqueeze(-1).expand(-1, -1, reshaped_layer_new_latents_with_metric.size(-1)).to(device)
            top_values_tensor = torch.gather(reshaped_layer_new_latents_with_metric, 1, expanded_top_indices).to(device)
        else:
            top_tokens_tensor = new_tokens_tensor_int.repeat(latents_tokens_int[layer_index].shape[0], 1, 1).to(device)
            top_values_tensor = reshaped_layer_new_latents_with_metric.to(device)
        
        
        combined_tokens = torch.cat((latents_tokens_int[layer_index], top_tokens_tensor), dim=1).to(device)
        combined_values = torch.cat((latents_values[layer_index].to(device), top_values_tensor), dim=1).to(device)
        metric_values = combined_values[:, :, 0].to(device)
        _, top_indices = torch.topk(metric_values, k=sequences_per_latent, dim=1)
        
        updated_tokens_tensor = torch.gather(combined_tokens, 1, top_indices.unsqueeze(-1).expand(-1, -1, 64)).to(device)
        updated_values_tensor = torch.gather(combined_values, 1, top_indices.unsqueeze(-1).expand(-1, -1, 65)).to(device)

        latents_tokens[layer_index] = updated_tokens_tensor.to(torch.uint16).to(device)
        latents_values[layer_index] = updated_values_tensor.to(device)



def update_latents_from_next_latents(latents_positions, latents_values, new_latents):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    latents_positions_int =  latents_positions.clone().detach().to(torch.int32).to(device)
    new_latents_tensor = torch.stack(new_latents).to(device)
    latents_size = new_latents_tensor.shape[-1]
    
    new_latents_tensor_sequence_means = new_latents_tensor.mean(dim=2)
    
    curr_latents_tensor, next_latents_indices = new_latents_tensor_sequence_means.topk(min(sequences_per_latent, new_latents_tensor.shape[1]), dim=1)
    next_latents_indices = next_latents_indices[1:].permute(0, 2, 1).unsqueeze(-1).to(device)
    next_latents_tensor = new_latents_tensor_sequence_means[:-1].unsqueeze(1).expand(-1, latents_size, -1, -1).to(device)
    
    next_latents_tensor, next_latents_positions = torch.topk(torch.gather(next_latents_tensor, 2, next_latents_indices.expand(-1, -1, -1, latents_size)), k=latents_per_sequence, dim=-1)
    
    curr_latents_tensor = curr_latents_tensor[:-1].permute(0, 2, 1).unsqueeze(-1).abs().to(device)
    
    new_latents_values = torch.cat((curr_latents_tensor, next_latents_tensor), dim=3).to(device)
    
    updated_latents_values = torch.cat((latents_values.to(device), new_latents_values), dim=2)
    
    updated_top, updated_indices = torch.topk(updated_latents_values[..., 0], k=sequences_per_latent, dim=-1)

    updated_latents_values = torch.gather(updated_latents_values, dim=2, index=updated_indices.unsqueeze(-1).expand(-1, -1, -1, latents_per_sequence+1))
    
    next_latents_positions =  next_latents_positions.to(torch.int32).to(device)
    updated_latents_positions = torch.cat((latents_positions_int, next_latents_positions), dim=2)
    updated_latents_positions = torch.gather(updated_latents_positions, dim=2, index=updated_indices.unsqueeze(-1).expand(-1, -1, -1, latents_per_sequence))
    
    for i in range(len(new_latents)-1):
        latents_positions[i] = updated_latents_positions[i]
        latents_values[i] = updated_latents_values[i]

    if latents_size == 1024:
        unique_positions = list(set(latents_positions[0][0][0].tolist()) ^ set(latents_positions[0][200][0].tolist()))
        print("  |  Num Unique Latents Sample:", len(unique_positions), end="")



def update_latents_from_prev_latents(latents_positions, latents_values, new_latents):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    latents_positions_int =  latents_positions.clone().detach().to(torch.int32).to(device)
    new_latents_tensor = torch.stack(new_latents).to(device)
    latents_size = new_latents_tensor.shape[-1]
    
    new_latents_tensor_sequence_means = new_latents_tensor.mean(dim=2)
    
    curr_latents_tensor, prev_latents_indices = new_latents_tensor_sequence_means.topk(min(sequences_per_latent, new_latents_tensor.shape[1]), dim=1)
    prev_latents_indices = prev_latents_indices[:-1].permute(0, 2, 1).unsqueeze(-1).to(device)
    prev_latents_tensor = new_latents_tensor_sequence_means[1:].unsqueeze(1).expand(-1, latents_size, -1, -1).to(device)
    
    prev_latents_tensor, prev_latents_positions = torch.topk(torch.gather(prev_latents_tensor, 2, prev_latents_indices.expand(-1, -1, -1, latents_size)), k=latents_per_sequence, dim=-1)
    
    curr_latents_tensor = curr_latents_tensor[:-1].permute(0, 2, 1).unsqueeze(-1).abs().to(device)
    
    new_latents_values = torch.cat((curr_latents_tensor, prev_latents_tensor), dim=3).to(device)
    
    updated_latents_values = torch.cat((latents_values.to(device), new_latents_values), dim=2)
    
    updated_top, updated_indices = torch.topk(updated_latents_values[..., 0], k=sequences_per_latent, dim=-1)

    updated_latents_values = torch.gather(updated_latents_values, dim=2, index=updated_indices.unsqueeze(-1).expand(-1, -1, -1, latents_per_sequence+1))
    
    prev_latents_positions =  prev_latents_positions.to(torch.int32).to(device)
    updated_latents_positions = torch.cat((latents_positions_int, prev_latents_positions), dim=2)
    updated_latents_positions = torch.gather(updated_latents_positions, dim=2, index=updated_indices.unsqueeze(-1).expand(-1, -1, -1, latents_per_sequence))
    
    for i in range(len(new_latents)-1):
        latents_positions[i] = updated_latents_positions[i]
        latents_values[i] = updated_latents_values[i]



def collect_latents(turing, sae_models, shard_index, shard_count):

    shard = np.load(f"{dataset_path}/shard_{shard_index}.npy")
    shard = np.split(shard, np.where(shard == -1)[0])
    shard = [sequence[sequence != -1] for sequence in shard if sequence.size > 1]
    
    turing_batch_size = 256
    
    
    activations = [False for _ in range(TuringLLMConfig.n_layer)]
    
    for i in range(0, len(shard), turing_batch_size):
        start_time = time.time()
        
        total_batches_length = (len(shard) + turing_batch_size - 1) // turing_batch_size
        print(f"Turing Inference  |  Shard {shard_index + 1} / {shard_count}  |  Batch {str(i // turing_batch_size + 1).zfill(len(str(total_batches_length)))} / {total_batches_length}  |  ", end="")
        
        batch = shard[i:i + turing_batch_size]
        batch = [sequence.tolist() for sequence in batch]
        results, latents = turing.generate_batch(batch, max_length=65, tokenize=False, decode=False, ignore_end=True)
        
        # Latents Dim 1: MLP Down, Residual Stream
        # Latents Dim 2: Layers
        # Latents Dim 3: Batch
        # Latents Dim 4: Sequence
        
        start_update_latents_time = time.time()
        
        
        
        # MLP Down and Residual Stream
        
        ## From Sequence
        update_latents_from_sequence(latents_avg_residuals_tokens_from_sequence, latents_avg_residuals_values_from_sequence, batch, latents[1], "avg")
        update_latents_from_sequence(latents_avg_mlp_down_tokens_from_sequence, latents_avg_mlp_down_values_from_sequence, batch, latents[0], "avg")
        
        ## From Next Latents
        update_latents_from_next_latents(latents_residuals_latents_from_next_latents, latents_residuals_values_from_next_latents, latents[1])
        update_latents_from_next_latents(latents_mlp_down_latents_from_next_latents, latents_mlp_down_values_from_next_latents, latents[0])
        
        ## From Prev Latents
        update_latents_from_prev_latents(latents_residuals_latents_from_prev_latents, latents_residuals_values_from_prev_latents, latents[1])
        update_latents_from_prev_latents(latents_mlp_down_latents_from_prev_latents, latents_mlp_down_values_from_prev_latents, latents[0])
        
        
        
        # SAE Latents
        process_sae_latents(sae_models, batch, latents[0])
        
        
                            
        print(f"  |  Duration: {time.time()-start_time:.2f}s  |  Update Latents Duration: {time.time()-start_update_latents_time:.2f}s")
    
    
    


def run():
    current_path = os.path.dirname(__file__)
    os.makedirs(latents_path, exist_ok=True)
    
    shard_start = 207
    
    shard_count = get_shard_count()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    turing = TuringLLMForInference(collect_latents=True, max_length=65)
    
    sae_models = {}
    for layer_index in range(TuringLLMConfig.n_layer):
        k = 128 + (layer_index * 16)
        sae = SAE(TuringLLMConfig.n_embd, sae_dim, k, only_encoder=True).to(device)
        sae = torch.compile(sae)
        sae.load(f"SAE/sae/sae_layer_{layer_index}.pth")
        sae_models[str(layer_index)] = sae
        
    if shard_count > 1:
        # Load From Sequence
        latents_avg_residuals_tokens_from_sequence = torch.from_numpy(np.load(latents_path + "/latents_avg_residuals_tokens_from_sequence.npy")).to(torch.uint16)
        latents_avg_residuals_values_from_sequence = torch.load(latents_path + "/latents_avg_residuals_values_from_sequence.pth")
        latents_avg_mlp_down_tokens_from_sequence = torch.from_numpy(np.load(latents_path + "/latents_avg_mlp_down_tokens_from_sequence.npy")).to(torch.uint16)
        latents_avg_mlp_down_values_from_sequence = torch.load(latents_path + "/latents_avg_mlp_down_values_from_sequence.pth")
        latents_avg_sae_tokens_from_sequence = torch.from_numpy(np.load(latents_path + "/latents_avg_sae_tokens_from_sequence.npy")).to(torch.uint16)
        latents_avg_sae_values_from_sequence = torch.load(latents_path + "/latents_avg_sae_values_from_sequence.pth")

        # Load From Next Latents
        latents_residuals_latents_from_next_latents = torch.from_numpy(np.load(latents_path + "/latents_residuals_latents_from_next_latents.npy")).to(torch.uint16)
        latents_residuals_values_from_next_latents = torch.load(latents_path + "/latents_residuals_values_from_next_latents.pth")
        latents_mlp_down_latents_from_next_latents = torch.from_numpy(np.load(latents_path + "/latents_mlp_down_latents_from_next_latents.npy")).to(torch.uint16)
        latents_mlp_down_values_from_next_latents = torch.load(latents_path + "/latents_mlp_down_values_from_next_latents.pth")

        # Load From Prev Latents
        latents_residuals_latents_from_prev_latents = torch.from_numpy(np.load(latents_path + "/latents_residuals_latents_from_prev_latents.npy")).to(torch.uint16)
        latents_residuals_values_from_prev_latents = torch.load(latents_path + "/latents_residuals_values_from_prev_latents.pth")
        latents_mlp_down_latents_from_prev_latents = torch.from_numpy(np.load(latents_path + "/latents_mlp_down_latents_from_prev_latents.npy")).to(torch.uint16)
        latents_mlp_down_values_from_prev_latents = torch.load(latents_path + "/latents_mlp_down_values_from_prev_latents.pth")
        latents_sae_latents_from_prev_latents = torch.from_numpy(np.load(latents_path + "/latents_sae_latents_from_prev_latents.npy")).to(torch.uint16)
        latents_sae_values_from_prev_latents = torch.load(latents_path + "/latents_sae_values_from_prev_latents.pth")
        
        print("")
        print("latents_avg_residuals_values_from_sequence", [values[0] for values in latents_avg_residuals_values_from_sequence[0][0].tolist()])
        print("latents_avg_mlp_down_values_from_sequence ", [values[0] for values in latents_avg_mlp_down_values_from_sequence[0][0].tolist()])
        print("latents_avg_sae_values_from_sequence      ", [values[0] for values in latents_avg_sae_values_from_sequence[0][0].tolist()])
        print("")
        print("latents_mlp_down_values_from_next_latents", [str("{:.4f}".format(values[0])) + " - " + str("{:.2f}".format(values[1])) for values in latents_mlp_down_values_from_next_latents[0][0].tolist()])
        print("latents_mlp_down_values_from_prev_latents", [str("{:.4f}".format(values[0])) + " - " + str("{:.2f}".format(values[1])) for values in latents_mlp_down_values_from_prev_latents[0][0].tolist()])
        print("latents_sae_values_from_prev_latents     ", [str("{:.4f}".format(values[0])) + " - " + str("{:.2f}".format(values[1])) for values in latents_sae_values_from_prev_latents[0][0].tolist()])
        print("")
        print("")
    
    for shard_index in range(shard_count)[shard_start-1:]:
        start_time = time.time()
        collect_latents(turing, sae_models, shard_index, shard_count)
        
        # Save From Sequence
        np.save(latents_path + "/latents_avg_residuals_tokens_from_sequence.npy", latents_avg_residuals_tokens_from_sequence.numpy())
        torch.save(latents_avg_residuals_values_from_sequence, latents_path + "/latents_avg_residuals_values_from_sequence.pth")
        np.save(latents_path + "/latents_avg_mlp_down_tokens_from_sequence.npy", latents_avg_mlp_down_tokens_from_sequence.numpy())
        torch.save(latents_avg_mlp_down_values_from_sequence, latents_path + "/latents_avg_mlp_down_values_from_sequence.pth")
        np.save(latents_path + "/latents_avg_sae_tokens_from_sequence.npy", latents_avg_sae_tokens_from_sequence.numpy())
        torch.save(latents_avg_sae_values_from_sequence, latents_path + "/latents_avg_sae_values_from_sequence.pth")
        
        # Save From Next Latents
        np.save(latents_path + "/latents_residuals_latents_from_next_latents.npy", latents_residuals_latents_from_next_latents.numpy())
        torch.save(latents_residuals_values_from_next_latents, latents_path + "/latents_residuals_values_from_next_latents.pth")
        np.save(latents_path + "/latents_mlp_down_latents_from_next_latents.npy", latents_mlp_down_latents_from_next_latents.numpy())
        torch.save(latents_mlp_down_values_from_next_latents, latents_path + "/latents_mlp_down_values_from_next_latents.pth")
        # np.save(latents_path + "/latents_sae_latents_from_next_latents.npy", latents_sae_latents_from_next_latents.numpy())
        # torch.save(latents_sae_values_from_next_latents, latents_path + "/latents_sae_values_from_next_latents.pth")
        
        # Save From Prev Latents
        np.save(latents_path + "/latents_residuals_latents_from_prev_latents.npy", latents_residuals_latents_from_prev_latents.numpy())
        torch.save(latents_residuals_values_from_prev_latents, latents_path + "/latents_residuals_values_from_prev_latents.pth")
        np.save(latents_path + "/latents_mlp_down_latents_from_prev_latents.npy", latents_mlp_down_latents_from_prev_latents.numpy())
        torch.save(latents_mlp_down_values_from_prev_latents, latents_path + "/latents_mlp_down_values_from_prev_latents.pth")
        np.save(latents_path + "/latents_sae_latents_from_prev_latents.npy", latents_sae_latents_from_prev_latents.numpy())
        torch.save(latents_sae_values_from_prev_latents, latents_path + "/latents_sae_values_from_prev_latents.pth")
        
        if shard_index % 12 == 0:
            print("Backing Up...")
            print("")
            print("")
            os.makedirs("latents-backup", exist_ok=True)
            shutil.copytree(current_path + "/" + latents_path, current_path + "/latents-backup-temp")
            with open(current_path + "/latents-backup-temp/save.txt", "w") as f:
                f.write(f"shard_index {str(shard_index)}")
            shutil.rmtree(current_path + "/latents-backup")
            os.rename("latents-backup-temp", "latents-backup")
        
        print("")
        print("latents_avg_residuals_values_from_sequence", [values[0] for values in latents_avg_residuals_values_from_sequence[0][0].tolist()])
        print("latents_avg_mlp_down_values_from_sequence ", [values[0] for values in latents_avg_mlp_down_values_from_sequence[0][0].tolist()])
        print("latents_avg_sae_values_from_sequence      ", [values[0] for values in latents_avg_sae_values_from_sequence[0][0].tolist()])
        print("")
        print("latents_mlp_down_values_from_next_latents", [str("{:.4f}".format(values[0])) + " - " + str("{:.2f}".format(values[1])) for values in latents_mlp_down_values_from_next_latents[0][0].tolist()])
        print("latents_mlp_down_values_from_prev_latents", [str("{:.4f}".format(values[0])) + " - " + str("{:.2f}".format(values[1])) for values in latents_mlp_down_values_from_prev_latents[0][0].tolist()])
        print("latents_sae_values_from_prev_latents     ", [str("{:.4f}".format(values[0])) + " - " + str("{:.2f}".format(values[1])) for values in latents_sae_values_from_prev_latents[0][0].tolist()])
        print("")
        print(f"Shard Duration: {time.time()-start_time:.2f}s")
        print("")
        print("")



run()


