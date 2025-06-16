from torch import Tensor, nn
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim (int):    dimension of the input and output token embeddings
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 3 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.Dropout(0.2))
        self.norm = nn.LayerNorm(embedding_dim)
    
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
        
