import os
import math
import random
import time
import datetime
import pytz
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

from model import TuringLLM
from tokenizer import Tokenizer
from hellaswag_eval import Hellaswag



batch_size = 8
resume_model = ""
model_name = "Turing-LLM-1.0-254M"



@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5



tokenizer = Tokenizer()
eos_token_id = tokenizer.get_eos_token()




class DataLoader:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        self.split = split
        self.shard_paths = []
        self.last_part_paths = []
        self.complete = False
        
        # Get Shard Filenames
        self.dataset_parts = ["val", "p1-e1", "p1-e2", "p2-e1"]
        dataset_path = "../synthetic-dataset/llm_training_dataset"
        dataset_path_contents = os.listdir(dataset_path)
        for dataset_part in self.dataset_parts:
            if dataset_part not in dataset_path_contents:
                raise Exception("LLM Training Dataset Not Found")
        
        for i, dataset_part in enumerate(self.dataset_parts):
            new_shards = os.listdir(dataset_path + "/" + dataset_part)
            new_shards = [shard for shard in new_shards if split in shard]
            new_shards = [os.path.join(dataset_path + "/" + dataset_part, shard) for shard in new_shards]
            self.shard_paths = self.shard_paths + new_shards
            if i == len(self.dataset_parts) - 1:
                self.last_part_paths = new_shards
        
        if len(self.shard_paths) == 0:
            raise Exception(f"Shards not found for {split} split")
        
        self.reset()
        
    def reset(self):
        self.current_shard = 0
        self.load_tokens(self.shard_paths[self.current_shard], False)
        self.current_position = 0
        
    def load_tokens(self, filename, remainder_tokens):
        tokens_np = np.load(filename)
        new_tokens = torch.tensor(tokens_np, dtype=torch.long)
        if remainder_tokens is not False:
            self.tokens = torch.cat((remainder_tokens, new_tokens))
        else:
            self.tokens = new_tokens
        
    def next_batch(self, silent=False):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        input_pos = torch.arange(T).expand(B, T)
        
        self.current_position += B * T
        
        # Handle Next Batch
        if self.current_position + (B * T + 1) > len(self.tokens):
            if silent is False and self.complete is False:
                print("")
                print("Getting Next Batch from Next Shard: ", self.shard_paths[(self.current_shard + 1) % len(self.shard_paths)])
                print("")
                
            remainder_length = (self.current_position + (B * T + 1)) - len(self.tokens)
            remainder_tokens = self.tokens[-remainder_length:] if remainder_length > 0 else False
            
            if self.complete is False:
                self.current_shard = (self.current_shard + 1) % len(self.shard_paths)
            if self.split == "train" and self.current_shard == 0 and self.current_position != 0 and self.split == "train":
                self.complete = True
            
            if self.complete is False:
                self.load_tokens(self.shard_paths[self.current_shard], remainder_tokens)
            else:
                self.current_shard = -1
                self.load_tokens(self.last_part_paths[self.current_shard], remainder_tokens)
            
            self.current_position = 0
            
            if self.split == "train" and self.current_shard == 0:
                if silent is False:
                    print("")
                    print("Getting Next Batch from Next Epoch")
                    print("")
            
        return x, y, input_pos
    
    def get_state(self):
        return { 'current_position': self.current_position, 'current_shard': self.current_shard }
    
    def set_state(self, current_position, current_shard):
        self.current_position = current_position
        self.current_shard = current_shard



device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(12)
if device == 'cuda':
    torch.cuda.manual_seed(12)

total_batch_size = 524288
B = batch_size
T = TuringLLMConfig().block_size
grad_accum_steps = total_batch_size // (B * T)

train_loader = DataLoader(B=B, T=T, split="train")
val_loader = DataLoader(B=B, T=T, split="val")

torch.set_float32_matmul_precision('high')



# Turing LLM Training Dataset
token_count = 2088960785
max_steps = (token_count // (B*T*grad_accum_steps)) + 2 # 3986
warmup_steps = 350
special_run_interval = int(max_steps / 64) # 276
special_run_save_interval = special_run_interval * 4



start_training_time = time.time()



log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_{math.floor(start_training_time)}.txt")

with open(log_file, "w") as f:
    pass

def write_to_log(text):
    with open(log_file, "a") as f:
        f.write(text)



models_dir = "models"
os.makedirs(f"{models_dir}/model_{math.floor(start_training_time)}", exist_ok=True)



def load_and_resume_training(checkpoint_path, device):
    if checkpoint_path is False:
        return False, False, False

    checkpoint = torch.load(checkpoint_path, map_location=device)

    random.setstate(checkpoint['rng_states']['python'])
    np.random.set_state(checkpoint['rng_states']['numpy'])
    torch.set_rng_state(torch.ByteTensor(checkpoint['rng_states']['torch']))
    if torch.cuda.is_available() and checkpoint['rng_states']['cuda'] is not None:
        torch.cuda.set_rng_state_all([torch.ByteTensor(s) for s in checkpoint['rng_states']['cuda']])
    
    model = TuringLLM(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model = torch.compile(model)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device=device)
    optimizer.load_state_dict(checkpoint['optimizer'])

    step = checkpoint['step']

    return model, step, optimizer



# Initialize Model
if len(resume_model) != 0 and resume_model is not False:
    # Load Model & Resume Training
    checkpoint_path = os.path.join(models_dir, resume_model)
    model, resume_step, optimizer = load_and_resume_training(checkpoint_path, device)
    resume_step += 1
    print("Resuming Training...")
    for _ in range(resume_step * grad_accum_steps):
        x, y, _ = train_loader.next_batch(silent=True)
    
    # Open Previous Log File
    prev_log_file = log_dir + "/log_" + resume_model.split("/", maxsplit=1)[0].split("model_")[1] + ".txt"
    with open(log_file, "a") as f:
        with open(prev_log_file, "r") as f_prev:
            prev_log_text = f_prev.read().strip()
            last_hella_score = prev_log_text.split("hella")[-1].split("\n")[0]
            prev_log_text = "hella".join(prev_log_text.split("hella")[:-1]) + "hella" + last_hella_score
            f.write(prev_log_text.strip() + "\n")
else:
    # Initialize New Model
    model = TuringLLM(TuringLLMConfig())
    model.to(device)
    model = torch.compile(model)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device=device)
    resume_step = 0



# Learning Rate
max_lr = 6e-4 # 6e-4 * 3
min_lr = max_lr * 0.1

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)



# Benchmarks
hellaswag = Hellaswag()





# ---------------------------------------------------
# Training
# ---------------------------------------------------



print("")
print("Model Name:        ", str(model_name))
print("Run Name:          ", f"model_{math.floor(start_training_time)}")
print("Current Time:      ", datetime.datetime.now(tz=pytz.timezone("Europe/London")))
print("")
print("Parameters Count:  ", '{:,}'.format(sum(p.numel() for p in model.parameters())))
print("Trainable Params:  ", '{:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
print("Max Steps:         ", str(max_steps))
print("")
print("")
print("")



for step in range(resume_step, max_steps):
    t0 = time.time()
    
    
    
    # Training
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y, input_pos = train_loader.next_batch()
        x, y, input_pos = x.to(device), y.to(device), input_pos.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y, input_pos)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
    training_steps_percent = ((step+1)/max_steps) * 100
    print(" | ".join([
        f"Step {str(step+1).zfill(len(str(max_steps)))} ({training_steps_percent:.2f}%)",
        f"Loss: {str(loss_accum.item())[:8]}",
        f"lr: {lr:.4e}",
        "Norm: " + str("{0:.4f}".format(norm))[:6],
        f"Duration: {dt:.2f}ms",
        f"tok/sec: {tokens_per_sec:.2f}"
    ]))
    write_to_log(f"{step} train {loss_accum.item():.6f}\n")
    if dt > 80000:
        torch.cuda.empty_cache()
    
    
    
    # Special Runs
    if step == 0 or (step+1) % special_run_interval == 0 or step == max_steps - 1:
        
        
        
        # Validation
        print("")
        print("")
        print("")
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y, input_pos = val_loader.next_batch()
                x, y, input_pos = x.to(device), y.to(device), input_pos.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y, input_pos)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f"Validation Loss:    {val_loss_accum.item():.4f}")
        write_to_log(f"{step} val {val_loss_accum.item():.4f}\n")



        # Hellaswag
        num_correct_norm, num_total, acc_norm = hellaswag.evaluate(model, device)
        print(f"HellaSwag Accuracy: {acc_norm:.4f} ({num_correct_norm}/{num_total})")
        write_to_log(f"{step} hella {acc_norm:.4f}\n")
    
    
    
        # Print Samples
        model.eval()
        num_return_sequences = 3
        max_length = 64
        sample_texts = ["Physics", "Artificial Intelligence", "Hello"]
        print("")
        print("")
        for sample_text_index, sample_text in enumerate(sample_texts):
            print(f"Sample Text {sample_text_index + 1}")
            tokens = tokenizer.encode(sample_text)
            tokens = [eos_token_id] + tokens
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(12)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    logits, _ = model(xgen)
                    logits = logits[:, -1, :]
                    props = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(props, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)

            print("")
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = tokenizer.decode(tokens)
                print("")
                print(f"Sample Completion {i + 1}: ")
                print(decoded.replace("\n", " \\n ")[len("<|endoftext|>"):].strip())
                print("")
            print("")
            print("")
        print("")
        print("")
        
        torch.cuda.empty_cache()
    
    
    
    # Save Checkpoint
    if (step > 1 and (step+1) % special_run_save_interval == 0) or step == max_steps - 1:
        checkpoint_path = os.path.join(models_dir, f"model_{math.floor(start_training_time)}", f"model_{math.floor(start_training_time)}_{(step+1):05d}.pt")
        model.save_checkpoint(checkpoint_path, optimizer, step, max_steps, warmup_steps)
        print("Checkpoint Saved:  ", str(checkpoint_path))


    
    # Print Current Time & Next Special Step
    if step == 0 or (step+1) % special_run_interval == 0 or step == max_steps - 1:
        print("Current Time:      ", datetime.datetime.now(tz=pytz.timezone("Europe/London")))
        next_special_step_num = int(min(step + 1 + special_run_interval, max_steps))
        if step == 0:
            next_special_step_num += -1
        print("Next Special Step: ", str(next_special_step_num))
        print("")
        print("")
        print("")
        


print("")
end_training_time = time.time()
print("Training Duration: ", end_training_time - start_training_time)
