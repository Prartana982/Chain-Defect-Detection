import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from src.dataset import ChainDataset

class CAEn(nn.Module):
    def __init__(self):
        super(CAEn, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32x32
            nn.ReLU(),
             nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16x16
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), # 256x256
            nn.Sigmoid() # Output 0-1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transform (Resize to 256x256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor() # 0-1 range matches Sigmoid
    ])
    
    dataset = ChainDataset(root_dir=args.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = CAEn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"Training CAE on {len(dataset)} images for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for images, _, _ in dataloader:
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss/len(dataloader):.6f}")
        
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/train/good")
    parser.add_argument("--model_path", type=str, default="cae_model.pth")
    parser.add_argument("--epochs", type=int, default=50) # Autoencoders need more epochs
    args = parser.parse_args()
    train(args)
