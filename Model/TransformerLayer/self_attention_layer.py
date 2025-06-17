from torch import nn
from torch import Tensor

class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embedding_dim (int):    Dimension of input token embeddings, each Q, K, V vectors, and output of this layer
            num_heads (int):        Number of heads that Q, K, V are split between to learn different kinds of relationships
                                    between patches in parallel
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

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
        x_norm = self.norm(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        return x + attn_output