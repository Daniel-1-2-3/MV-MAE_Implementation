import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = int(embedding_dim / num_heads)
        
        self.qkv_projection = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor):
        # Get Q, K, V and distribute among the heads
        batch, num_patches, _ = x.size()
        qkv = self.qkv_projection(x)  # Q, K, V for all heads
        qkv = qkv.view(batch, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, num_patches, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # shape = (batch, num_heads, num_patches, head_dim)

        # Vectorized attention for all heads
        attention_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        attention_score = torch.matmul(attention_weights, V)  # shape = (batch, num_heads, num_patches, head_dim)
        final_attention = attention_score.transpose(1, 2).contiguous().view(batch, num_patches, -1)  # (batch, num_patches, embedding_dim)

        x = self.norm(x + final_attention)  # post add-norm
        return x
        