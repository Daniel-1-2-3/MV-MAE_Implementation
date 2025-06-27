import torch
import torch.nn as nn
from torch import Tensor
from Model.TransformerLayer.mlp_layer import FeedForward
from Model.TransformerLayer.self_attention_layer import MultiHeadSelfAttention
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim (int):    Dimension of input and output of the transformer block
            num_heads (int):    Number of heads in multi-headed self attention layer
        """
        super().__init__()
        self.mlp = FeedForward(embed_dim)
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
    
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
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
        for _ in range(12)])
    
    def forward(self, x):
        for block in self.encoder:
            x = block(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.decoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
        for _ in range(8)])
      
    def forward(self, x):
        for block in self.decoder:
            x = block(x)
        return x