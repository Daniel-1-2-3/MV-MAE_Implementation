import torch
from torch import nn
from torch import Tensor
import math, random

class PrepareEncoderInput(nn.Module):
    def __init__ (self, in_channels, total_patches, embed_dim, patch_size, training=True):
        """
        Each view is passed through a series of 4 convolutional layers (with shared weights across views) 
        to extract patch embeddings. Fixed sine/cosine positional embeddings are then added to each patch 
        to encode spatial information. View embeddings are also added. 

        Args:
            in_channels (int):  Typically 3 for RGB input
            img_size (int):     Side length of the input image
            patch_size (int):   Side length of each patch
            embed_dim (int):    Dimension of each patch token vector
            training (bool):    During training, one view is masked, and 25% of other 
                                view's patches are also masked. (default True)
        """
        super().__init__()
        
        self.total_patches = total_patches
        self.embed_dim = embed_dim
        self.training = training

        self.patch_extract = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0), 
            nn.GELU(), # (batch, embed * 2, img_size/patch_size, img_size/patch_size)
        )
        
        self.register_buffer("positional_embeds", 
            self.sin_cos_embed(int(math.sqrt(self.total_patches)), self.embed_dim))   
        self.view_embed = nn.Parameter(torch.randn(2, embed_dim))
         
        self.masked_ids = None
        self.partial_view, self.masked_view = None, None
        self.partial_id = None # String of "L"/"R"
    
    def sin_cos_embed(self, grid_size, embed_dim):
        """
        Generate sin/cos embeddings representing the 
        position of each patch token in the image (grid).

        Args:
            grid_size (int):    The number of patches determines the amount 
                                of positional embeddings that are needed
            embed_dim (int):    The dimension of each patch token determines 
                                the length of its positional embedding (how many individual 
                                sine and cosine components are used to encode its position.
        Returns:
            (Tensor):   Outputs a shape of (num_patches, embed_dim), 
                        able to be directly concatenated to the patch embedding
        """
        # Grid of patch positions
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (grid_size, grid_size, 2)
        grid = grid.reshape(-1, 2) # (total_patches, 2)

        # Apply sin/cos to x and y poses
        dim_half = embed_dim // 2
        term = torch.exp(torch.arange(0, dim_half, 2) * (-math.log(10000.0) / dim_half))
        pos_x = grid[:, 0].unsqueeze(1) * term # (num_patches, dim_half//2)
        pos_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=1)  # (num_patches, dim_half)
        pos_y = grid[:, 1].unsqueeze(1) * term
        pos_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=1)

        # Concat x and y embeddings
        pos_embed = torch.cat([pos_x, pos_y], dim=1) # (num_patches, embed_dim)
        return pos_embed

    def random_mask(self, x: Tensor, mask_ratio):
        batch, total_patches, embed_dim = x.shape
        num_keep = int(total_patches * (1 - mask_ratio))

        scores = torch.rand(batch, total_patches, device=x.device)
        ids_sorted = torch.argsort(scores, dim=1)
        ids_keep = ids_sorted[:, :num_keep]     
        ids_masked = ids_sorted[:, num_keep:] 

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim))  # (B, kept, D)
        return x_masked, ids_masked


    def forward(self, x1: Tensor, x2: Tensor):
        """
        Args:
            x1 (Tensor):    Tensor of first view in stereo vision (batch, channels, H, W)
            x2 (Tensor):    Tensor of second view in stereo vision (batch, channels, H, W)

        Returns:
            (Tensor):   Output is a tensor that includes all unmasked patches, with a
                        shape of (batch, num_unmasked_patches, embed_dim)
        """
        batch_size = x1.size(0)
        x1_clone, x2_clone = x1.clone(), x2.clone()
        
        x1 = self.patch_extract(x1) # (batch, embed_dim, img_size/patch_size, img_size/patch_size)
        x1 = x1.flatten(2, 3)
        x1 = x1.transpose(1, 2) # (batch, total_patches, embed_dim)
        x1 = x1 + self.positional_embeds.unsqueeze(0).to(x1.device)
        
        x2 = self.patch_extract(x2)
        x2 = x2.flatten(2, 3)
        x2 = x2.transpose(1, 2)
        x2 = x2 + self.positional_embeds.unsqueeze(0).to(x2.device)
        
        view_ids = torch.cat([
            torch.zeros(self.total_patches, dtype=torch.long, device=x1.device),
            torch.ones (self.total_patches, dtype=torch.long, device=x1.device)
        ], dim=0)  # (2*total_patches,)
        view_emb = self.view_embed[view_ids] # (2*total_patches, encoder_embed_dim)
        view_emb = view_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        x1 = x1 + view_emb[:, :self.total_patches, :]
        x2 = x2 + view_emb[:, self.total_patches:, :]
                
        x = None
        if self.training:
            if random.random() < 0.5:
                self.partial_view = x1_clone
                self.masked_view = x2_clone
                self.partial_id = "L"
                
                x, self.masked_ids = self.random_mask(x1, 0.25)
                full_masked_ids = torch.arange(self.total_patches, 2 * self.total_patches, device=x.device)
                self.masked_ids = torch.cat([self.masked_ids, full_masked_ids.unsqueeze(0).expand(x.size(0), -1)], dim=1)
            else:
                self.partial_view = x2_clone
                self.masked_view = x1_clone
                self.partial_id = "R"
                
                x, self.masked_ids = self.random_mask(x2, 0.25)
                full_masked_ids = torch.arange(0, self.total_patches, device=x.device)
                self.masked_ids = torch.cat([self.masked_ids + self.total_patches, full_masked_ids.unsqueeze(0).expand(x.size(0), -1)], dim=1)
        else:
            x = torch.cat((x1, x2), dim=1)
        
        return x
    
    def get_views(self):
        return self.partial_view, self.masked_view