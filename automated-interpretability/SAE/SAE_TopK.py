import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import einops



random.seed(12)



@torch.no_grad()
def geometric_median(points, max_iter: int = 200, tol: float = 1e-5):
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess
        weights = 1 / torch.norm(points - guess, dim=1)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break

    return guess



class SAE(nn.Module):
    
    def __init__(self, d_model, d_sae, k=100, only_encoder=False):
        super().__init__()
        
        self.k = k

        self.encoder = nn.Linear(d_model, d_sae)
        self.encoder.bias.data.zero_()

        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().T
        self.set_decoder_norm_to_unit_norm()

        self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_tokens_since_fired = torch.zeros(d_sae, dtype=torch.long, device=self.device)
        self.auxk_alpha = 1 / 32
        self.dead_feature_threshold = 10_000_000
        
        self.only_encoder = only_encoder

    def encode(self, x):
        post_relu_feat_acts = nn.functional.relu(self.encoder(x - self.decoder_bias))
        post_topk = post_relu_feat_acts.topk(self.k, sorted=False, dim=-1)

        top_acts = post_topk.values
        top_indices = post_topk.indices

        buffer = torch.zeros_like(post_relu_feat_acts)
        encoded_acts = buffer.scatter_(dim=-1, index=top_indices, src=top_acts)

        return encoded_acts, top_acts, top_indices

    def decode(self, x):
        return self.decoder(x) + self.decoder_bias

    def forward(self, x):
        latents, _, _ = self.encode(x)
        y = self.decode(latents)
        return y, latents

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.decoder.weight.dtype).eps
        norm = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.weight.grad is not None

        parallel_component = einops.einsum(
            self.decoder.weight.grad,
            self.decoder.weight.data,
            "d_in d_sae, d_in d_sae -> d_sae",
        )
        self.decoder.weight.grad -= einops.einsum(
            parallel_component,
            self.decoder.weight.data,
            "d_sae, d_in d_sae -> d_in d_sae",
        )
        
    def train_forward(self, x):
        latents, top_acts, top_indices = self.encode(x)
        y = self.decode(latents)
        
        e = y - x
        total_variance = (x - x.mean(0)).pow(2).sum(0)
        
        # effective_l0 = top_acts.size(1)
        
        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[top_indices.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0
        
        dead_mask = (
            self.num_tokens_since_fired > self.dead_feature_threshold
            if self.auxk_alpha > 0
            else None
        ).to(latents.device)
        dead_features = int(dead_mask.sum())
        
        # If dead features: Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], latents, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer = torch.zeros_like(latents)
            auxk_acts = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Encourage the top ~50% of dead latents to predict the residual of the top k living latents
            e_hat = self.decode(auxk_acts)
            auxk_loss = (e_hat - e).pow(2)  # .sum(0)
            auxk_loss = scale * torch.mean(auxk_loss / total_variance)
        else:
            auxk_loss = y.new_tensor(0.0)

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = auxk_loss.sum(dim=-1).mean()
        loss = l2_loss + self.auxk_alpha * auxk_loss
        
        return latents, loss, dead_features

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.num_tokens_since_fired = torch.load(".".join(path.split(".")[:-1]) + "_num_tokens_since_fired.pth").to(self.device)
        
        if self.only_encoder is True:
            del self.decoder

    def save(self, path):
        torch.save(self.state_dict(), path)
        torch.save(self.num_tokens_since_fired, ".".join(path.split(".")[:-1]) + "_num_tokens_since_fired.pth")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class SAETrainer():
    
    def __init__(self, sae, sae_layer_index, sae_max_layer_index, sae_path, batch_size=48488):
        self.sae = sae
        self.sae_layer_index = sae_layer_index
        self.sae_max_layer_index = sae_max_layer_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sae.to(self.device)
        self.batch_size = batch_size
        self.log_file = f"{sae_path}/log_sae_layer_{sae_layer_index}.txt"
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                    pass

    def train(self, data, step_offset=0):
        lr_max = 2e-4
        lr_max += (3 * lr_max * (self.sae_layer_index / self.sae_max_layer_index))
        optimizer = optim.AdamW(self.sae.parameters(), lr=lr_max, betas=(0.9, 0.999))
        
        max_steps = 26000
        half_lr_start = 500
        decay_start = 1000
        second_jump_lr_start = 3000
        third_jump_lr_start = 13442
        
        def lr_lambda(step):
            if (step + step_offset) < half_lr_start:
                return 1.0
            if (step + step_offset) < decay_start:
                return 0.5
            if (step + step_offset) < second_jump_lr_start:
                return ((max_steps - (step + step_offset)) / (max_steps - decay_start)) * 0.5
            if (step + step_offset) < third_jump_lr_start:
                return ((max_steps - (step + step_offset)) / (max_steps - decay_start)) * 0.5 * 0.25
            return ((max_steps - (step + step_offset)) / (max_steps - decay_start)) * 0.5 * 0.25 * 0.5

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


        self.sae.train()
        optimizer.zero_grad()
        
        data_length = len(data) - (len(data) % self.batch_size)
        data = data[:data_length]
        random.shuffle(data)

        # Training Loop
        adj_step = step_offset
        for step in range(0, len(data), self.batch_size):
            adj_step = (step / self.batch_size) + step_offset
            
            batch = torch.stack(data[step:step+self.batch_size])
            x = batch.to(self.device)
            
            if step == 0:
                print("Initializing Decoder Bias")
                median = geometric_median(x)
                self.sae.decoder_bias.data = median
                
            self.sae.set_decoder_norm_to_unit_norm()
            
            latents, loss, dead_features = self.sae.train_forward(x)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
            self.sae.remove_gradient_parallel_to_decoder_directions()
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            lr = optimizer.param_groups[0]["lr"]
            
            total_steps = int((len(data) / self.batch_size) + step_offset)
            if int(adj_step + 1) == total_steps or int(adj_step + 1) % 4 == 0:
                print(f"SAE Training  |  Layer {str(self.sae_layer_index + 1).zfill(len(str(self.sae_max_layer_index + 1)))} / {self.sae_max_layer_index + 1}  |  Step {str(int(adj_step + 1)).zfill(len(str(total_steps)))} / {total_steps}  |  lr: {lr:.2e}  |  Loss: {loss.item():.6f}")

            with open(self.log_file, "a") as f:
                f.write(f"total_loss {loss} dead_features {dead_features}\n")
            
        return adj_step
