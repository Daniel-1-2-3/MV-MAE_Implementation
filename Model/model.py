from Model.TransformerLayer.transformer_block import Encoder, Decoder
from Model.prepare_encoder_in import PrepareEncoderInput
from Model.prepare_decoder_in import PrepareDecoderInput

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from pytorch_msssim import ssim
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_channels=3,
                 encoder_embed_dim=768, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_num_heads=8, 
                 training=True):
        
        super().__init__()
        self.img_size, self.patch_size, self.in_channels = img_size, patch_size, in_channels
        self.total_patches = int((self.img_size / patch_size) ** 2)
        
        self.prepare_encoder_in = PrepareEncoderInput(
            in_channels=3, total_patches=self.total_patches, 
            embed_dim=encoder_embed_dim, training=training,
        )
        self.encoder = Encoder(
            embed_dim=encoder_embed_dim, num_heads=encoder_num_heads
        )
        
        self.prepare_decoder_in = PrepareDecoderInput(
            total_patches=self.total_patches, 
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
        )
        self.decoder = Decoder(
            embed_dim=decoder_embed_dim, num_heads=decoder_num_heads
        )
        self.output_projection = nn.Linear(decoder_embed_dim, in_channels * patch_size ** 2)
        self.reconstructed_1, self.reconstructed_2 = None, None
    
    def get_reconstructed_imgs(self, x: Tensor):
        """
        Args:
            x (Tensor): Output of the decoder with shape of
                        (batch_size, num_patches_both_views, decoder_embed_dim)
        Returns:
            img1, img2: (Tensor), (Tensor) both of shape (batch_size, in_channels, img_size, img_size), 
                        the correct format for rgb images
        """
        x = torch.sigmoid(self.output_projection(x)) # Normalizes output between (0, 1)
        
        batch_size, patches_both_imgs, _ = x.shape
        patches_per_img = patches_both_imgs // 2
        grid_size = self.img_size // self.patch_size
        
        decoded_view1 = x[:, :patches_per_img, :]
        decoded_view1 = decoded_view1.reshape(
            batch_size, grid_size, grid_size, self.patch_size, 
            self.patch_size, self.in_channels)
        img1 = decoded_view1.permute(0, 5, 1, 3, 2, 4)
        img1 = img1.reshape(batch_size, self.in_channels, self.img_size, self.img_size)
        
        decoded_view2 = x[:, patches_per_img:, :]
        decoded_view2 = decoded_view2.reshape(
            batch_size, grid_size, grid_size, self.patch_size, 
            self.patch_size, self.in_channels)
        img2 = decoded_view2.permute(0, 5, 1, 3, 2, 4)
        img2 = img2.reshape(batch_size, self.in_channels, self.img_size, self.img_size)
        
        return img1, img2
    
    def get_loss(self, decoder_output):
        """
        Calculate MSE loss and perceptual loss (ssim) of both reconstructed views, 
        with a final loss being the average of their losses

        Args:
            decoder_output (Tensor):    (batch, num_patches_both_views, decoder_embed_dim), 
                                        which gets passed through self.get_reconstructed_imgs
        Returns:
            total_loss (int): Total loss, comprised of an equal weighting of MSE and SSIM loss
        """
        self.reconstructed_1, self.reconstructed_2 = self.get_reconstructed_imgs(decoder_output)
        ref_partial_view, ref_masked_view = self.prepare_encoder_in.get_views() # (batch, in_channels, img_size, img_size)
        
        # MSE loss per pixel
        mse_loss1 = torch.clamp(F.mse_loss(self.reconstructed_1, ref_partial_view), min=0.0)
        mse_loss2 = torch.clamp(F.mse_loss(self.reconstructed_2, ref_masked_view), min=0.0)
        mse_loss = (mse_loss1 + mse_loss2) / 2
        
        # Calculate SSIM perceptual loss
        ssim_loss1 = 1 - torch.clamp(ssim(self.reconstructed_1, ref_partial_view, data_range=1.0), min=0.0)
        ssim_loss2 = 1 - torch.clamp(ssim(self.reconstructed_2, ref_masked_view, data_range=1.0), min=0.0)
        ssim_loss = (ssim_loss1 + ssim_loss2) / 2
        
        total_loss = 0.5 * mse_loss + 0.5 * ssim_loss
        return total_loss

    def render_reconstructed(self):
        """
        Renders the reconstructed partial and masked views, taking only 
        the first pair in the batch.
        """
        package = zip([self.reconstructed_1, self.reconstructed_2], 
                      ["Reconstructed Partial View", "Reconstructed Masked View"])
        for i, (img, title) in enumerate(package):
            plt.subplot(1, 2, i + 1)
            plt.imshow(img[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.title(title)
            plt.axis("off")
            
        plt.tight_layout()
        plt.show()
    
    def forward(self, x1: Tensor, x2: Tensor):
        """
        Full model architecture 
        Args:
            x1 (Tensor): First view (batch, in_channels, img_size, img_size)
            x2 (Tensor): Second view (same shape as x1)

        Returns:
            (Tensor): Output of decoder (batch, num_patches_both_views, decoder_embed_dim)
        """
        x = self.prepare_encoder_in(x1, x2)
        x = self.encoder(x)
        x = self.prepare_decoder_in(x, 
            self.prepare_encoder_in.visible_ids, 
            self.prepare_encoder_in.partial_view_id
        )
        x = self.decoder(x)
        return x 
        