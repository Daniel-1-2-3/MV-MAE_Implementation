from torch import nn, Tensor
import torch
import einops
import torch.nn.functional as F

from Model.encoder import ViTMaskedEncoder
from Model.decoder import ViTMaskedDecoder
from Model.prepare_input import Prepare

class Model(nn.Module):
    def __init__(self, 
            nviews=2,
            patch_size=8,
            encoder_embed_dim=768,
            decoder_embed_dim=512,
            encoder_heads=16,
            decoder_heads=16,
            in_channels=3,
            img_h_size=128,
            img_w_size=128, 
        ):
        super().__init__()
        self.nviews = nviews
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.in_channels = in_channels
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size
        self.img_w_fused = self.nviews * self.img_w_size
        self.num_patches = int((self.img_h_size * self.img_w_size) // (patch_size ** 2) * nviews)
        
        self.encoder = ViTMaskedEncoder(depth=4)
        self.decoder = ViTMaskedDecoder(depth=2)
        self.out_proj = nn.Linear(decoder_embed_dim, self.num_patches ** 2 * in_channels)
    
    def forward(self, x: Tensor):
        """
        Whole pipeline of the MV-MAE model: patchified, 
        then passed through encoder, mask tokens added, and 
        passed through decoder. 

        Args:
            x (Tensor): Representing all the views stitched together horizontally,
                        with a shape of (batch, height, width_total, channels)

        Returns:
            x (Tensor): (batch, total_patches, patch_size^2 * channels)
        """
        x, mask = self.encoder(x)
        x = self.decoder(x, mask)
        out = self.out_proj(x)
        return out
    
    def compute_loss(self, out, truth, mask):
        """
        Compute MSE loss

        Args:
            out (_type_): (batch, total_patches, patch_size^2 * channels)
            truth (_type_): Fused views (batch, height, width_total, channels)
            mask (_type_):  Has shape (batch, total_num_patches), where each vector in the 
                            last dimension is a binary mask with 0 representing unmasked, and 
                            1 representing masked

        Returns:
            loss (Tensor): MSE loss, retains grad information
        """
        truth = self.patchify_ground_truth_views(truth)
        loss_per_patch = F.mse_loss(out, truth).mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum() # Only calculate loss for masked patches
        return loss

    def patchify_ground_truth_views(self, x: Tensor):
        """
        Convert the ground truth views into patches to match the format of the
        decoder output, in order to compute loss

        Args:
            x (Tensor): Representing all the views stitched together horizontally,
                        with a shape of (batch, height, width_total, channels)
        
        Returns:
            x (Tensor): (batch, total_patches, patch_size^2 * channels)
        """
        batch, height, w_total, in_channels = x.shape
        assert w_total % self.nviews == 0, "Width must be divisible by number of views"

        # Split along width into views into (b, h, w, c) x nviews
        views = torch.chunk(x, self.nviews, dim=2)  # List of [b, h, w, c]
        patchified_views = [ # Rearrange
            einops.rearrange(v, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', 
                p1=self.patch_size, p2=self.patch_size) for v in views
        ]

        return torch.cat(patchified_views, dim=1)


