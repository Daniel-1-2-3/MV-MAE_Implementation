import torch
from torch import Tensor, nn

class PrepareDecoderInput(nn.Module):
    def __init__(self, total_patches, encoder_embed_dim, decoder_embed_dim):
        """
        Args:
            total_patches (int):        The number of patches in a single, unmasked view
            encoder_embed_dim (int):    Embedding dimension of each patch token outputted 
                                        by the encoder
            decoder_embed_dim (int):    Embedding dimension of each patch token fed into the 
                                        decoder, smaller than that of encoder
        """
        super().__init__()
        
        self.decoder_embed_dim = decoder_embed_dim
        self.change_dim = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        
        self.mask_tokens = nn.Parameter(
            torch.randn(1, total_patches * 2, decoder_embed_dim))
        
        self.learnable_pos_embeds = nn.Parameter(torch.randn(1, 2 * total_patches, decoder_embed_dim))
        self.learnable_view_embeds = nn.Parameter(torch.randn(1, 2 * total_patches, decoder_embed_dim))

    
    def forward(self, x: Tensor, visible_ids: Tensor):
        """
        This layer prepares the encoder's output for input into the decoder. Learnable
        mask tokens are inserted into places where the patch was masked during encoding, 
        resulting in a Tensor of shape (batch, 2 * total_patches, decoder_embed_dim), since it
        is comprised learnable mask token placeholders and encoder output tokens, together 
        representing both left and right views. Positional embeddings and view 
        embeddings are added.

        Args:
            x (Tensor):             The output of the encoder, with a shape
                                    of (batch, num_unmasked_patches, encoder_embed_dim)
            visible_ids (Tensor):   Ids representing the positions of the unmasked patches 
                                    in the image grid, with a shape of (batch_size, num_unmasked_patches)

        Returns:
            (Tensor) with a shape of (batch_size, 2 * total_patches, decoder_embed_dim)
        """
        
        batch_size = x.shape[0]
        x = self.change_dim(x)
        
        # Insert unmasked patches at their positions
        tokens = self.mask_tokens.expand(batch_size, -1, -1).clone()  # IMPORTANT: clone so it’s not shared!
        x_full = tokens.clone() 
        for b in range(batch_size):
            x_full[b].index_copy_(0, visible_ids[b], x[b])
            
        learnable_pos_embeds = self.learnable_pos_embeds.expand(batch_size, -1, -1).to(x.device)
        learnable_view_embeds = self.learnable_view_embeds.expand(batch_size, -1, -1).to(x.device)
        x_full = x_full + learnable_pos_embeds + learnable_view_embeds
        
        return x_full