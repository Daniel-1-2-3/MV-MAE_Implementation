from Model.model import Model

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from pytorch_msssim import ssim
from PIL import Image
from torchvision import transforms
import os

if __name__ == "__main__":
    
    left_image_path = os.path.join(os.getcwd(), 'Dataset', 'Val', 'LeftCam', 'left_20.png')
    left_img = Image.open(left_image_path).convert("RGB")
    right_image_path = os.path.join(os.getcwd(), 'Dataset', 'Val', 'RightCam', 'right_20.png')
    right_img = Image.open(right_image_path).convert("RGB")

    img_size = 128
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    x1 = transform(left_img).unsqueeze(0) # (batch, channels, img_size, img_size)
    x2 = transform(right_img).unsqueeze(0) 

    model = Model()
    
    with torch.no_grad():
        decoder_output = model(x1, x2)
        loss = model.get_loss(decoder_output)
        print("Loss:", loss)
    
    model.render_reconstructed()