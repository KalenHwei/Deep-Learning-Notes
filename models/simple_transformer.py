import torch
import torch.nn as nn
import torch.nn.functional as F
from lora.lora import LoRALinear

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, lora_r=0, lora_alpha=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, lora_r=lora_r, lora_alpha=lora_alpha)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout, lora_r=lora_r, lora_alpha=lora_alpha)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention部分
        src2 = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(src2))

        # FFN部分
        src2 = self.feed_forward(src)
        src = self.norm2(src + self.dropout(src2))

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1, lora_r=0, lora_alpha=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout, lora_r=lora_r, lora_alpha=lora_alpha)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)



class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention（decoder自身的输入）
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # Cross-attention（与encoder输出交互）
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # FFN
        tgt2 = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)

# 以下为 MultiHeadAttention 和 FeedForward 的简单实现
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, lora_r=0, lora_alpha=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须为 num_heads 的整数倍"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = LoRALinear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha)
        self.k_linear = LoRALinear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha)
        self.v_linear = LoRALinear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha)
        self.out_proj = LoRALinear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha)

    def forward(self, query, key, value, mask=None):
        B, T, C = query.size()
        # 分头计算
        q = self.q_linear(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = attn @ v
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1, lora_r=0, lora_alpha=0.0):
        super().__init__()
        self.linear1 = LoRALinear(embed_dim, hidden_dim, lora_r=lora_r, lora_alpha=lora_alpha)
        self.linear2 = LoRALinear(hidden_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
