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
        
        self.partial_view_mask_tokens = nn.Parameter(
            torch.randn(1, total_patches, decoder_embed_dim))
        self.masked_view_mask_tokens = nn.Parameter(
            torch.randn(1, total_patches, decoder_embed_dim))
        self.learnable_pos_embeds = nn.Parameter(
            torch.randn(1, 2 * total_patches, decoder_embed_dim))
        
        self.view1_embed = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=False)
        self.view2_embed = nn.Parameter(torch.ones(1, 1, decoder_embed_dim), requires_grad=False)
    
    def forward(self, x: Tensor, visible_ids: Tensor, partial_view_id):
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
            partial_view_id (int):  The view id (0 or 1) of the non-hidden view

        Returns:
            (Tensor) with a shape of (batch_size, 2 * total_patches, decoder_embed_dim)
        """
        
        batch_size = x.shape[0]
        x = self.change_dim(x)
        
        partial_view_mask_tokens = self.partial_view_mask_tokens.expand(batch_size, -1, -1)
        partial_view = partial_view_mask_tokens.scatter(1, visible_ids.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim), x)
        masked_view = self.masked_view_mask_tokens.expand(batch_size, -1, -1)
        
        if partial_view_id == 0:
            partial_view += self.view1_embed
            masked_view += self.view2_embed
        else:
            partial_view += self.view2_embed
            masked_view += self.view1_embed
            
        x = torch.cat([partial_view, masked_view], dim=1)
        
        learnable_pos_embeds = self.learnable_pos_embeds.expand(batch_size, -1, -1)
        x += learnable_pos_embeds
        
        return x
        
        
        