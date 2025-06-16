import torch.nn as nn
from TransformerLayer.mlp_layer import FeedForward
from TransformerLayer.self_attention_layer import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        """
        Args:
            embedding_dim (int):    dimension of input and output of the transformer block
            num_heads (int):        number of heads in multi-headed self attention layer
        """
        super().__init__()
        self.mlp = FeedForward(embedding_dim)
        self.self_attention = MultiHeadSelfAttention(embedding_dim, num_heads)
    
    def forward(self, x):
        """
        Implements residual connections between self attention layers and feed forward
        layers in the transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_patches, embedding_dim)
        Returns:
            (Tensor):   Output shape is the same as input
        """
        x = x + self.self_attention(x)
        x = x + self.mlp(x)
        return x