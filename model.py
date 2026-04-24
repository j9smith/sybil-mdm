import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from params import MAX_SEQ_LENGTH, T_EMBEDDING_DIM

device = 'cuda'
dtype = torch.float32

t_freqs = torch.exp(
    -math.log(10000.0)
    * torch.arange(T_EMBEDDING_DIM // 2, device=device, dtype=dtype)
    / (T_EMBEDDING_DIM // 2)
)

position_indices = torch.arange(MAX_SEQ_LENGTH)

def sinusoidal_positional_encoding(timesteps: torch.Tensor):
    """Sinusoidal to encode diffusion process progression"""
    args = timesteps[:, None] * t_freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

def rotary_positional_encoding(q: torch.Tensor, k: torch.Tensor, rope_freqs):
    """
    RoPE to encode token position in sequence according to 
    eq. 34 in RoFormer: Enhanced Transformer with Rotary Position Embedding
    (Su et al, 2023).
    """
    def _rotate(x):
        seq_len = x.shape[-2]
        angles = position_indices[:seq_len, None].to(x.device) * rope_freqs[None, :] # mθ
        angles = angles.repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
        
        x2 = x.reshape(*x.shape[:-1], -1, 2) # Group x into pairs [x1, x2, ...] -> #[[x1, x2], ...]
        x2 = x2.flip(-1) # -> [[x2, x1], ...]
        x2 = x2 * torch.tensor([-1, 1], device=x.device) # [[-x2, x1], ...]
        x2 = x2.reshape(*x.shape) # [-x2, x1, ...]
        
        return (x * torch.cos(angles)) + (x2 * torch.sin(angles))

    E_q = _rotate(q)
    E_k = _rotate(k)
    
    return E_q, E_k

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network as described in 'GLU Variants Improve Transformer'
    (Shazeer, 2020).

    (Swish1(xW1) ⊗ xV)W2
    """
    def __init__(self, input_dims: int, ffn_hidden_dims: int):
        super().__init__()
        self.xW = nn.Linear(input_dims, ffn_hidden_dims)
        self.xV = nn.Linear(input_dims, ffn_hidden_dims)
        self.out = nn.Linear(ffn_hidden_dims, input_dims)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.out(self.act(self.xW(x)) * self.xV(x))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, rope_freqs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.qkv = nn.Linear(
            in_features=d_model,
            out_features=d_model * 3
        )
        
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope_freqs = rope_freqs

    def forward(self, x: torch.Tensor):
        B, seq_len, d_model = x.shape

        q, k, v = torch.chunk(
            input=self.qkv(x),
            chunks=3,
            dim=-1
        )

        def _reshape(x):
            return x.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k, v = _reshape(q), _reshape(k), _reshape(v)
        E_q, E_k = rotary_positional_encoding(q, k, self.rope_freqs)

        out = F.scaled_dot_product_attention(
            query=E_q,
            key=E_k,
            value=v
        )

        out = out.transpose(1, 2)
        out = out.reshape(B, seq_len, d_model)

        return self.out_proj(out)

class TimeEmbedding(nn.Module):
    def __init__(self, base_dim, hidden_dim, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, t):
        return self.mlp(sinusoidal_positional_encoding(t))

class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block following Attention is All You Need (Vaswani et. al, 2017)
    and incorporating modern improvements (pre-norm, AdaLN).
    """
    def __init__(self, n_attn_heads, d_model, t_emb_dims, ffn_hidden_dims):
        super().__init__()
        self.ffn = SwiGLUFFN(input_dims=d_model, ffn_hidden_dims=ffn_hidden_dims)

        assert d_model % n_attn_heads == 0, f"d_model ({d_model}) is not divisible by n_attn_heads ({n_attn_heads})"
        head_dim = d_model // n_attn_heads

        rope_freqs = torch.exp(-math.log(10000.0) * torch.arange(head_dim // 2) / (head_dim // 2)).to("cuda")

        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=n_attn_heads, 
            head_dim=head_dim,
            rope_freqs=rope_freqs
        )

        self.norm1 = torch.nn.RMSNorm(normalized_shape=d_model)
        self.norm2 = torch.nn.RMSNorm(normalized_shape=d_model)

        self.ada_proj = nn.Linear(t_emb_dims, d_model * 4)

    def forward(self, x: torch.Tensor, E_t: torch.Tensor):
        """
        Forward pass uses pre-norm and AdaLN. 
        """
        scale1, shift1, scale2, shift2 = torch.chunk(self.ada_proj(E_t), chunks=4, dim=-1) # AdaLN

        h1 = self.norm1(x)
        h1 = h1 * (1 + scale1) + shift1
        h1 = self.attn(h1) + x

        h2 = self.norm2(h1)
        h2 = h2 * (1 + scale2) + shift2
        h2 = self.ffn(h2) + h1

        return h2

class SybilMDM(nn.Module):
    """
    Implementation of a masked diffusion model for language modelling, following
    Large Language Diffusion Models (Nie et. al, 2025).
    """
    def __init__(
            self,
            vocab_size,
            n_transformer_blocks: int = 4,
            d_model: int = 1024,
            n_attention_heads: int = 8,
            ffn_dims: int = 4096,
            t_emb_dims: int = 256,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.embed_t = TimeEmbedding(
            base_dim=T_EMBEDDING_DIM,
            hidden_dim=4 * T_EMBEDDING_DIM,
            embed_dim=t_emb_dims
        )

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                t_emb_dims=t_emb_dims,
                ffn_hidden_dims=ffn_dims,
                n_attn_heads=n_attention_heads,
            ) for _ in range(0, n_transformer_blocks)
        ])

        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, t):
        E_x = self.token_emb(x)
        E_t = self.embed_t(t)

        for block in self.blocks:
            E_x = block(E_x, E_t)

        return self.out_proj(E_x)