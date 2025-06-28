from CreateDataset.dataset import StereoImageDataset
from Model.encoder import ViTMaskedEncoder
from Model.prepare_input import Prepare
import os
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

def debug(out, model):
    print('Shape:', out.shape)
    print("Mean:", out.mean().item())
    print("Std dev:", out.std().item())
    print("NaNs:", torch.isnan(out).any().item())
    
    cos = torch.nn.functional.cosine_similarity(out[:, 1:, :], out[:, 1:, :].mean(dim=1, keepdim=True), dim=-1)
    print("Mean cosine similarity to average patch:", cos.mean().item())

    print("\n--- Weight Statistics ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_mean = param.data.mean().item()
            weight_std = param.data.std().item()
            weight_min = param.data.min().item()
            weight_max = param.data.max().item()
            print(f"{name}: mean={weight_mean:.4e}, std={weight_std:.4e}, min={weight_min:.4e}, max={weight_max:.4e}")
            
if __name__ == "__main__":
    encoder = ViTMaskedEncoder()
    
    base_dataset_dir=os.path.join(os.getcwd(), 'dataset')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
        
    dir = os.path.join(base_dataset_dir, 'Val')
    dataset = StereoImageDataset(root_dir=dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, 
        num_workers=8, pin_memory=True, drop_last=True)
    
    i = 0
    for epoch in range(0, 10):
        if i >= 1:
            break
        total_loss = 0.0
        for x1, x2 in tqdm(loader, desc=f"Epoch {epoch + 1} - Training", leave=False):
            if i >= 1:
                break
            x = Prepare.fuse_normalize([x1, x2])
            out, mask = encoder(x)
            loss = out.mean()  # dummy loss for debugging
            loss.backward()
            debug(out, encoder)
            i += 1
            