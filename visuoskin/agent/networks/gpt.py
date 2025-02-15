"""
An adaptation of Andrej Karpathy's nanoGPT implementation in PyTorch.
Original source: https://github.com/karpathy/nanoGPT

Original License:
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Original comments:
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class CrossAttention(nn.Module):
    def __init__(self, repr_dim, nhead=4, nlayers=4, use_buffer_token=False):
        super().__init__()
        self.tf_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=repr_dim,
                nhead=nhead,
                dim_feedforward=repr_dim * 4,
                batch_first=True,
            ),
            num_layers=nlayers,
        )
        if use_buffer_token:
            self.buffer_token = nn.Parameter(torch.randn(1, 1, repr_dim))
        self.use_buffer_token = use_buffer_token

    def forward(self, feat, cond):
        # add buffer token to the beginning of the sequence
        if self.use_buffer_token:
            batch_size = feat.size(0)
            buffer_token = self.buffer_token.expand(batch_size, 1, -1)
            cond_with_buffer = torch.cat([buffer_token, cond], dim=1)
            return self.tf_decoder(feat, cond_with_buffer)
        else:
            return self.tf_decoder(feat, cond)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            # torch.ones(1, 1, config.block_size, config.block_size),
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, attn_mask=None):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.bias[:, :, :T, :T]
        if attn_mask is not None:
            mask = mask * attn_mask
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y



import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryCausalSelfAttention(nn.Module):
  

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

    
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

        self.cached_k = []
        self.cached_v = []
        self.cached_video_names = []
        self.max_mem = 4

        kernel_size = 2
        stride = 2
        padding = kernel_size // 2
        self.compress_k = nn.Conv1d(
            in_channels=self.head_dim,  
            out_channels=self.head_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=self.head_dim,  
            bias=False,
        )
        self.norm_compress_k = nn.LayerNorm(self.head_dim)

        self.compress_v = nn.Conv1d(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=self.head_dim,
            bias=False,
        )
        self.norm_compress_v = nn.LayerNorm(self.head_dim)


        self.enable_compression = True

    def clear_memory(self):
        """Clear all cached memory (K/V). Useful at new episode / new video."""
        self.cached_k.clear()
        self.cached_v.clear()
        self.cached_video_names.clear()

    def compress_fn(self, k_uncompressed, v_uncompressed):
        
        B, nH, T, hdim = k_uncompressed.shape

        k_out = k_uncompressed.reshape(B * nH, T, hdim)
        k_out = k_out.permute(0, 2, 1)  # (B*nH, hdim, T)
        k_out = self.compress_k(k_out)  # => (B*nH, hdim, T_compressed)
        T_compressed = k_out.shape[-1]
        k_out = k_out.permute(0, 2, 1)  # => (B*nH, Tc, hdim)
        k_out = self.norm_compress_k(k_out)
        k_out = k_out.reshape(B, nH, T_compressed, hdim)

        v_out = v_uncompressed.reshape(B * nH, T, hdim)
        v_out = v_out.permute(0, 2, 1)
        v_out = self.compress_v(v_out)
        T_compressed = v_out.shape[-1]
        v_out = v_out.permute(0, 2, 1)
        v_out = self.norm_compress_v(v_out)
        v_out = v_out.reshape(B, nH, T_compressed, hdim)

        return k_out, v_out

    def forward(self, x, attn_mask=None, video_names=None):


        B, T, C = x.size()

        qkv = self.c_attn(x)  # => (B, T, 3C)
        q, k_new, v_new = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # => (B, n_head, T, head_dim)
        k_new = k_new.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v_new = v_new.view(B, T, self.n_head, self.head_dim).transpose(1, 2)


        if len(self.cached_video_names) > 0:

            old_vids = self.cached_video_names[-1]
            if video_names is not None and old_vids is not None:
                for i in range(B):
                    if video_names[i] not in old_vids:
                        self.clear_memory()
                        break

        if self.enable_compression and len(self.cached_k) > 0:

            k_uncompressed = self.cached_k[-1]
            v_uncompressed = self.cached_v[-1]
            k_compressed, v_compressed = self.compress_fn(k_uncompressed, v_uncompressed)

            self.cached_k[-1] = k_compressed.detach()
            self.cached_v[-1] = v_compressed.detach()

        self.cached_k.append(k_new.detach())
        self.cached_v.append(v_new.detach())

        self.cached_video_names.append(video_names)

        
        while len(self.cached_k) > self.max_mem:
            self.cached_k.pop(0)
            self.cached_v.pop(0)
            self.cached_video_names.pop(0)

        
        all_k = torch.cat(self.cached_k, dim=2)
        all_v = torch.cat(self.cached_v, dim=2)


        att = (q @ all_k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))


        T_mem = all_k.size(2)
        max_size = self.bias.shape[-1]
        used_T = min(T, max_size)
        used_T_mem = min(T_mem, max_size)

        causal_mask = self.bias[:, :, :used_T, :used_T_mem]
        if used_T < T or used_T_mem < T_mem:
          
            att = att[:, :, :used_T, :used_T_mem]

        if attn_mask is not None:
            
            attn_mask = attn_mask[:, :, :used_T, :used_T_mem]
            causal_mask = causal_mask * attn_mask

        att = att.masked_fill(causal_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att) 

        all_v = all_v[:, :, :used_T_mem, :]
        y = att @ all_v

        y = y.transpose(1, 2).contiguous().view(B, used_T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# class Block(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.ln_1 = nn.LayerNorm(config.n_embd)
#         self.attn = CausalSelfAttention(config)
#         self.ln_2 = nn.LayerNorm(config.n_embd)
#         self.mlp = MLP(config)

#     def forward(self, x, mask=None):
#         x = x + self.attn(self.ln_1(x), attn_mask=mask)
#         x = x + self.mlp(self.ln_2(x))
#         return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        
        self.attn = MemoryCausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, mask=None, video_names=None):
        x = x + self.attn(self.ln_1(x), attn_mask=mask, video_names=video_names)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    input_dim: int = 256
    output_dim: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.input_dim is not None
        assert config.output_dim is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(config.input_dim, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def forward(self, input, targets=None, mask=None,video_names=None):
        device = input.device
        b, t, d = input.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            input
        )  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            
            x = block(x, mask=mask,video_names=video_names)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas)
        return optimizer