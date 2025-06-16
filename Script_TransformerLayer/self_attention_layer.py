import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, embedding_dim, num_heads):
        """
        Args:
            embedding_dim (int):    dimension of input token embeddings, each Q, K, V vectors, and output of this layer
            num_heads (int):        number of heads that Q, K, V are split between to learn different kinds of relationships
                                    between patches in parallel
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = int(embedding_dim / num_heads)
        
        self.qkv_projection = nn.Sequential(
            nn.Linear(embedding_dim, 3 * embedding_dim),
            nn.Dropout(0.2))
        self.out_projection = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor):
        """
        The multi-headed self attention layer uses a linear projection layer to calculate
        Q (Query), K (key), V (value). Each Q, K, V are split among several heads,
        where attention is calculated, representing how much focus each patch token 
        should give to all other tokens in the sequence. Attention from each head is combined 
        using a projection layer.

        Args:
            x (Tensor): Input tensor of shape (batch, num_patches, embedding_dim)

        Returns:
            (Tensor):   Output shape is the same as input
        """
        residual = x
        x = self.norm(x)
        
        # Get Q, K, V and distribute among the heads
        batch, num_patches, _ = x.size()
        qkv = self.qkv_projection(x) # (batch, num_patches, Q_dim + K_dim + V_dim)
        qkv = qkv.view(batch, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) 
        Q, K, V = qkv[0], qkv[1], qkv[2]  # (batch, num_heads, num_patches, head_dim)

        # Vectorized attention for all heads
        attention_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        attention_score = torch.matmul(attention_weights, V)
        final_attention = attention_score.transpose(1, 2).contiguous().view(batch, num_patches, -1)
        final_attention = self.out_projection(final_attention)
        
        return residual + final_attention
        