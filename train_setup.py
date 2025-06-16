
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, shutil

from Model.model import Model
from CreateDataset.dataset import StereoImageDataset

class Trainer():
    def __init__(self, img_size, patch_size, batch_size, in_channels,
                 encoder_embed_dim, encoder_num_heads,
                 decoder_embed_dim, decoder_num_heads, 
                 base_dataset_dir):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(
            img_size, patch_size, in_channels, encoder_embed_dim, 
            encoder_num_heads, decoder_embed_dim, decoder_num_heads, True)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()])
        
        print("Loading dataset...")
        train_dir, val_dir = os.path.join(base_dataset_dir, 'Train'), os.path.join(base_dataset_dir, 'Val')
        train_dataset = StereoImageDataset(root_dir=train_dir, transform=self.transform)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        
        val_dataset = StereoImageDataset(root_dir=val_dir, transform=self.transform)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        results_folder = os.path.join(os.getcwd(), 'results')
        if os.path.exists(results_folder):
            shutil.rmtree(results_folder)
        
    def train(self, num_epochs, lr):
        self.model.to(self.device)
        self.model.train()
        print(f'Running on {self.device}')
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
        for epoch in range(0, num_epochs):
            total_loss = 0.0
            for x1, x2 in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False):
                optimizer.zero_grad()
                
                x1, x2 = x1.to(self.device), x2.to(self.device)
                out = self.model(x1, x2)
                loss: torch.Tensor = self.model.get_loss(out)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(self.train_loader)
            avg_val_loss = self.evaluate()
            loss_log = f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}'
            self.update_and_save_losses(loss_log)
            
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(os.getcwd(), 'results', 'checkpoint.pt'))
            except:
                print('Failed to save checkpoint')
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x1, x2 in self.val_loader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                out = self.model(x1, x2)
                loss = self.model.get_loss(out)
                total_loss += loss.item()
                
        self.model.train()
        return total_loss / len(self.val_loader)
    
    def update_and_save_losses(self, loss_log):
        try:
            os.makedirs('results', exist_ok=True)
            log_path = os.path.join(os.getcwd(), 'results', "losses.txt")
            with open(log_path, "a") as f:
                f.write(loss_log + "\n")
        except:
            print('Failed to save losses')
            
        print(loss_log)