from Model.TransformerLayer.transformer_block import Encoder, Decoder
from Model.prepare_encoder_in import PrepareEncoderInput
from Model.prepare_decoder_in import PrepareDecoderInput

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as torchvision_F

class Model(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_channels=3,
                 encoder_embed_dim=768, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_num_heads=8, 
                 training=True, mse=0.2, ssim=0.8, debug = False):
        
        super().__init__()
        self.debug = debug
        self.mse, self.ssim = mse, ssim
        self.img_size, self.patch_size, self.in_channels = img_size, patch_size, in_channels
        self.total_patches = int((self.img_size / patch_size) ** 2)
        
        self.prepare_encoder_in = PrepareEncoderInput(
            in_channels=3, total_patches=self.total_patches, 
            embed_dim=encoder_embed_dim, patch_size=patch_size, training=training,
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
             
        self.register_buffer('mean', torch.tensor([0.51905, 0.47986, 0.48809]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.17454, 0.20183, 0.19598]).view(1, 3, 1, 1))
        
    def get_loss(self, x: Tensor):
        """
            Calculate MSE loss of reconstructed patches in both views.

            Args:
                decoder_output (Tensor):    (batch, num_patches_both_views, decoder_embed_dim), 
                                            which gets passed through self.get_reconstructed_imgs
            Returns:
                total_loss (int): Total loss, comprised of an equal weighting of MSE and SSIM loss
        """
        x = torch.sigmoid(self.output_projection(x))
        img1, img2 = self.get_reconstructed_imgs(x)
       
        ref_partial_view, ref_masked_view = self.prepare_encoder_in.get_views()
        ref_partial_view = ref_partial_view * self.std + self.mean
        ref_masked_view = ref_masked_view * self.std + self.mean

        mse_loss = F.mse_loss(torch.cat([img1, img2], 1), 
                    torch.cat([ref_partial_view, ref_masked_view], 1))
        
        ssim1 = ssim(img1, ref_partial_view, data_range=1.0) # both are +- 3 range, thus range = 6.0
        ssim2 = ssim(img2, ref_masked_view, data_range=1.0)
        ssim_loss = 1.0 - 0.5 * (ssim1 + ssim2)    
            
        total_loss = self.mse * mse_loss + self.ssim * ssim_loss
        return total_loss
            
    def get_reconstructed_imgs(self, x: Tensor):
        """
        Args:
            x (Tensor): Output of the decoder with shape of
                        (batch_size, num_masked_patches, decoder_embed_dim)
        Returns:
            img1, img2: (Tensor), (Tensor) both of shape (batch_size, in_channels, img_size, img_size), 
                        the correct format for rgb images
        """
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
        
        self.reconstructed_1, self.reconstructed_2 = img1, img2
        
        return img1, img2

    def render_reconstructed(self):
        """
        Renders the reconstructed partial and masked views, taking only 
        the first pair in the batch.
        """
        def unnormalize(tensor, mean = [0.51905, 0.47986, 0.48809], 
                        std = [0.17454, 0.20183, 0.19598]):
            mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
            std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
            return tensor * std + mean
                
        package = None
        if self.prepare_encoder_in.partial_id == "L":
            package = zip([self.reconstructed_1, self.reconstructed_2], 
                ["Reconstructed Partial View", "Reconstructed Masked View"])
        else:
            package = zip([self.reconstructed_2, self.reconstructed_1], 
                ["Reconstructed Partial View", "Reconstructed Masked View"])

        for i, (img, title) in enumerate(package):
            plt.subplot(1, 2, i + 1)
            img = unnormalize(img[0].cpu().detach(), mean=[0.51905, 0.47986, 0.48809], std=[0.17454, 0.20183, 0.19598])
            plt.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
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
        if self.debug:
            print('ENCODE PREP:', self.similarity(x))
        x = self.encoder(x)
        if self.debug:
            print('ENCODE:', self.similarity(x))
        x = self.prepare_decoder_in(x, self.prepare_encoder_in.masked_ids)
        if self.debug:
            print('DECODE PREP:', self.similarity(x))
        x = self.decoder(x)
        if self.debug:
            print('DECODE:', self.similarity(x))
        return x 

    def similarity(self, x: torch.Tensor): # For debug
        batch_size, num_patches, dim = x.shape
        x = F.normalize(x, dim=2)
        x = x.permute(1, 0, 2)
        patch_embeddings = x.mean(dim=1)  # (patches, dim)
        sim_matrix = patch_embeddings @ patch_embeddings.T  # (patches, patches)
        num = num_patches * (num_patches - 1)
        mean_sim = sim_matrix.masked_fill(torch.eye(num_patches, device=x.device).bool(), 0).sum() / num

        return mean_sim.item()
                