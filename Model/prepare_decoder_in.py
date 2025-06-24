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
        
        self.total_patches = total_patches
        self.decoder_embed_dim = decoder_embed_dim
        self.change_dim = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        
        self.learnable_pos_embeds = nn.Parameter(
            torch.randn(1, 2 * total_patches, decoder_embed_dim))
        self.view_embed = nn.Parameter(torch.randn(2, decoder_embed_dim))
    
    def forward(self, x: Tensor, masked_ids: Tensor):
        """
        This layer prepares the encoder's output for input into the decoder. Learnable mask 
        tokens are inserted, with the mask token placeholders and encoder output tokens together 
        representing both left and right views. Positional embeddings and view 
        embeddings are added.

        Args:
            x (Tensor):             The output of the encoder, with a shape
                                    of (batch, num_unmasked_patches, encoder_embed_dim)
            masked_ids (Tensor):   Ids representing the positions of the unmasked patches 
                                    in the image grid, with a shape of (batch_size, num_unmasked_patches)
        Returns:
            (Tensor) with a shape of (batch_size, 2 * total_patches, decoder_embed_dim)
        """
        
        batch_size = x.shape[0]
        x = self.change_dim(x)
        
        full_ids = torch.arange(2 * self.total_patches, device=masked_ids.device)
        visible_ids = []
        for b in range(masked_ids.size(0)):
            mask = torch.ones(2 * self.total_patches, dtype=torch.bool, device=masked_ids.device)
            mask[masked_ids[b]] = False 
            visible_ids.append(full_ids[mask]) 
        visible_ids = torch.stack(visible_ids, dim=0) # (batch, num_visible_ids)
        
        view_tokens = self.mask_token.expand(batch_size, 2 * self.total_patches, -1) # Random vals of shape (batch, patches across both imgs, decoder_embed_dim), later insert patch embeds into it
        view = view_tokens.scatter(1, visible_ids.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim), x)
        
        view_ids = torch.cat([
            torch.zeros(self.total_patches, dtype=torch.long, device=x.device),
            torch.ones (self.total_patches, dtype=torch.long, device=x.device)
        ], dim=0)  # (2*total_patches,)
        view_embed = self.view_embed[view_ids] # (2*total_patches, encoder_embed_dim)
        view_embed = view_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        learnable_pos_embeds = self.learnable_pos_embeds.expand(batch_size, -1, -1).to(x.device)
        
        x = view + learnable_pos_embeds + view_embed
        return x