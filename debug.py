from CreateDataset.dataset import StereoImageDataset
from Model.prepare_input import Prepare
from Model.model import Model
import os
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

def debug(out, model, file_path='debug.txt'):
    with open(file_path, 'a') as f:
        f.write('Shape: {}\n'.format(out.shape))
        f.write("Mean: {}\n".format(out.mean().item()))
        f.write("Std dev: {}\n".format(out.std().item()))
        f.write("NaNs: {}\n".format(torch.isnan(out).any().item()))
        
        cos = torch.nn.functional.cosine_similarity(out[:, 1:, :], out[:, 1:, :].mean(dim=1, keepdim=True), dim=-1)
        f.write("Mean cosine similarity to average patch: {}\n".format(cos.mean().item()))

        f.write("\n--- Weight Statistics ---\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_mean = param.data.mean().item()
                weight_std = param.data.std().item()
                weight_min = param.data.min().item()
                weight_max = param.data.max().item()
                f.write(f"{name}: mean={weight_mean:.4e}, std={weight_std:.4e}, min={weight_min:.4e}, max={weight_max:.4e}\n")
        f.write("\n")
            
if __name__ == "__main__":
    model = Model()
    base_dataset_dir=os.path.join(os.getcwd(), 'dataset')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
        
    dir = os.path.join(base_dataset_dir, 'Val')
    dataset = StereoImageDataset(root_dir=dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, 
        num_workers=8, pin_memory=True, drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003) 
    model.train()
    for epoch in range(0, 10):
        total_loss = 0.0
        for x1, x2 in tqdm(loader, desc=f"Epoch {epoch + 1} - Training", leave=False):
            optimizer.zero_grad()
            x = Prepare.fuse_normalize([x1, x2])
            out, mask = model(x)
            loss = model.compute_loss(out, x, mask)
            loss.backward()
            optimizer.step()
            model.render_reconstruction(out)
            debug(out, model)
            i += 1
            