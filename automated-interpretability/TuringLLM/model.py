import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchtune.modules import RMSNorm
import inspect



class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
    def forward(self, x, input_pos=None):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.c_proj(y)
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()        
        self.up_proj_swish = nn.Linear(config.n_embd, config.hidden_size)
        self.silu = nn.SiLU()
        self.up_proj = nn.Linear(config.n_embd, config.hidden_size)
        self.down_proj = nn.Linear(config.hidden_size, config.n_embd)
        self.down_proj.WEIGHT_SCALE_INIT = 1
        
    def forward(self, x):
        swish = self.silu(self.up_proj_swish(x))
        Vx = self.up_proj(x)
        x = self.down_proj(swish * Vx)
        return x



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)
        
    def forward(self, x, input_pos):
        x = x + self.attn(self.norm_1(x), input_pos)
        mlp_x = self.mlp(self.norm_2(x))
        x = x + mlp_x
        return x, mlp_x



class TuringLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm_f=RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        self.collect_latents = False
        self.max_length = 72
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'WEIGHT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None, input_pos=None):
        # B (Batch), T (Tokens)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length {T}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        latents_mlp_down = []
        latents_residuals = []
        for block_index, block in enumerate(self.transformer.h):
            x, mlp_x = block(x, input_pos)
            if self.collect_latents is True and T >= self.max_length - 1:
                latents_mlp_down.append(mlp_x)
                latents_residuals.append(x) # [:, -1, :]
        x = self.transformer.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        if self.collect_latents is True and T >= self.max_length - 1:
            latents_mlp_down = [latents.detach().cpu() for latents in latents_mlp_down]
            latents_residuals = [latents.detach().cpu() for latents in latents_residuals]
            return logits, latents_mlp_down, latents_residuals
        elif self.collect_latents is True:
            return logits, [], []
        
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    def save_checkpoint(self, checkpoint_path, optimizer, step, max_steps, warmup_steps):
        checkpoint = {
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': self.config,
            'step': step,
            'max_steps': max_steps,
            'warmup_steps': warmup_steps,
            'rng_states': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state().cpu().numpy(),
                'cuda': [s.cpu().numpy() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
            }
        }
        torch.save(checkpoint, checkpoint_path)
