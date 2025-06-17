from torch import Tensor, nn
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int):    Dimension of the input and output token embeds
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 3 * embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(3 * embed_dim, embed_dim),
            nn.Dropout(0.2))
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: Tensor):
        """
        FeedForward MLP block consisting of two linear projections with GELU activation and dropout.
        Layer normalization is used. Residual connection preserves output diversity. 

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_patches, embedding_dim)
        Returns:
            (Tensor):   Output shape is the same as input
        """
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        return residual + x  
        