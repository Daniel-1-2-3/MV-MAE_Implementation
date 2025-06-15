from patch_embeddings import PatchEmbedding
from TransformerLayer.transformer_block import TransformerBlock
from decoder_input_prepare import DecoderInputPreparation

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import cv2
import argparse
import lpips
from pytorch_msssim import ssim

class Model(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_channels=3,
                 encoder_embed_dim=768, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_num_heads=8, loss_weighting=[1.0, 0.75, 0.50]):
        super().__init__()
        self.img_size, self.in_channels = img_size, in_channels
        self.patch_size = patch_size
        self.num_patches = int(img_size / patch_size)**2
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        
        self.patch_embed = PatchEmbedding(in_channels, patch_size, encoder_embed_dim, img_size)
        self.encoder = nn.ModuleList([
            TransformerBlock(encoder_embed_dim, encoder_num_heads) for _ in range(12)
        ])
        self.prepare_decoder_in = DecoderInputPreparation(img_size, patch_size, encoder_embed_dim, decoder_embed_dim)
        self.decoder = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads) for _ in range(4)
        ])
        self.reconstruct_projection = nn.Linear(self.decoder_embed_dim, self.in_channels * self.patch_size**2)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips_loss_fn = lpips.LPIPS(net='alex').eval().to(device)
        self.loss_weighting = loss_weighting
    
    def forward(self, x):
        x = self.patch_embed.forward(x)  # Get decoder outputs for masked tokens
        cos_sim = torch.nn.functional.cosine_similarity(
            x[:, :, None, :], x[:, None, :, :], dim=-1
        )
        mean_sim = cos_sim.mean().item()
        print(f"Cos sim after patch embed: {mean_sim:.20f}")
        
        for encoder_block in self.encoder:
            x = encoder_block(x)
            
        cos_sim = torch.nn.functional.cosine_similarity(
            x[:, :, None, :], x[:, None, :, :], dim=-1
        )
        mean_sim = cos_sim.mean().item()
        print(f"Cos sim after encoder: {mean_sim:.20f}")
        
        
        x = self.prepare_decoder_in.forward(x)
        cos_sim = torch.nn.functional.cosine_similarity(
            x[:, :, None, :], x[:, None, :, :], dim=-1
        )
        mean_sim = cos_sim.mean().item()
        print(f"Cos sim after prepare decoder in: {mean_sim:.20f}")
        
        
        for decoder_block in self.decoder:
            x = decoder_block(x)

        a = x[:, self.num_patches:, :]
        cos_sim = torch.nn.functional.cosine_similarity(
            a[:, :, None, :], a[:, None, :, :], dim=-1
        )
        mean_sim = cos_sim.mean().item()
        print(f"Cos sim after decoder: {mean_sim:.20f}")
        
        # shape = (batch_size, num_patches(both images), vector dimension of each patch embed)
        x = self.reconstruct_projection(x[:, self.num_patches:, :]) # shape = (batch_size, num_patches(decoded image), num_pixels in each patch * channels)
        
        
        return x 
    
    def get_loss(self, x, original_img, show = False):
        grid_size = self.img_size // self.patch_size # Num of patches along one dimension
        
        current_batch_size = x.shape[0]
        reconstructed = x.view(current_batch_size, grid_size, grid_size, self.patch_size, self.patch_size, self.in_channels)
        reconstructed = reconstructed.permute(0, 5, 1, 3, 2, 4).contiguous()
        reconstructed = reconstructed.view(current_batch_size, self.in_channels, self.img_size, self.img_size)
        
        if show:
            # Convert from (C, H, W) to (H, W, C), scale to 0–255, and convert to uint8
            rec_img = reconstructed[0].permute(1, 2, 0).cpu().numpy()
            gt_img = original_img[0].permute(1, 2, 0).cpu().numpy()

            rec_img = (rec_img * 255).clip(0, 255).astype(np.uint8)
            gt_img = (gt_img * 255).clip(0, 255).astype(np.uint8)

            combined = np.hstack((gt_img, rec_img))
            cv2.imshow("Ground Truth (Left) | Reconstructed (Right)", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Calculated MSE loss per pixel
        mse_loss = torch.clamp(F.mse_loss(reconstructed, original_img), min=0.0)
        # Calculate SSIM perceptual loss
        ssim_loss = 1 - torch.clamp(ssim(reconstructed, original_img, data_range=1.0), min=0.0)
        # LPIP perceptual loss
        reconstructed_lpips = 2 * reconstructed - 1 # normalize to [-1, 1]
        original_lpips = 2 * original_img - 1
        lpips_loss = self.lpips_loss_fn(reconstructed_lpips, original_lpips).mean()
        lpips_loss = torch.clamp(lpips_loss, min=0.0)
        
        return self.loss_weighting[0] * ssim_loss + self.loss_weighting[1] * lpips_loss + self.loss_weighting[2] * mse_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()
    
    left_image_path = os.path.join(os.getcwd(), 'Dataset', 'Val', 'LeftCam', 'left_20.png')
    left_img = Image.open(left_image_path).convert("RGB")
    right_image_path = os.path.join(os.getcwd(), 'Dataset', 'Val', 'RightCam', 'right_20.png')
    right_img = Image.open(right_image_path).convert("RGB")

    img_size = 128
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    visible_tensor = transform(left_img).unsqueeze(0) # (batch, channels, img_size, img_size)
    masked_tensor = transform(right_img).unsqueeze(0) 

    model = Model(
        img_size=128,
        patch_size=8,
        in_channels=3,
        encoder_embed_dim=384,
        encoder_num_heads=6,
        decoder_embed_dim=256,
        decoder_num_heads=4,
        loss_weighting=(0.5, 1.5),
    )
    
    model.eval()
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(os.path.join('Results', args.weights), map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    
    with torch.no_grad():
        decoder_output = model(visible_tensor)
        loss = model.get_loss(decoder_output, masked_tensor, show=True)
        print("Loss:", loss.item())

        

    