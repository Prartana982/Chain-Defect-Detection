import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pickle
import os
from src.dataset import ChainDataset

class PaDiM:
    def __init__(self, backbone_name="resnet50", d_reduced=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_reduced = d_reduced # Dimension reduction using random selection
        
        # Load backbone
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.layers_to_extract = ['layer1', 'layer2', 'layer3'] # PaDiM typically uses more scales
        
        self.backbone.to(self.device)
        self.backbone.eval()
        
        self.features = []
        self._register_hooks()
        
        self.mean = None
        self.cov = None
        self.idx = None # Selected indices for dimensionality reduction

    def _register_hooks(self):
        def hook(module, input, output):
            self.features.append(output)
        for name, module in self.backbone.named_children():
            if name in self.layers_to_extract:
                module.register_forward_hook(hook)

    def _embed(self, images):
        self.features = []
        with torch.no_grad():
            _ = self.backbone(images.to(self.device))
        
        # Resize to match largest spatial resolution (Layer 1 usually)
        ref_h, ref_w = self.features[0].shape[2], self.features[0].shape[3]
        
        embeddings = []
        for feat in self.features:
            feat = F.interpolate(feat, size=(ref_h, ref_w), mode='bilinear', align_corners=True)
            embeddings.append(feat)
            
        embedding = torch.cat(embeddings, dim=1) 
        return embedding

    def fit(self, dataloader):
        print("Extracting features for PaDiM...")
        embedding_list = []
        
        for images, _, _ in tqdm(dataloader):
            # (B, C, H, W)
            emb = self._embed(images)
            embedding_list.append(emb.cpu())
            
        full_embeddings = torch.cat(embedding_list, dim=0) # (N, C, H, W)
        
        # Random dimensionality reduction
        _, C, H, W = full_embeddings.shape
        if self.idx is None:
            if C > self.d_reduced:
                self.idx = torch.randperm(C)[:self.d_reduced]
            else:
                self.idx = torch.arange(C)
        
        subset_embeddings = full_embeddings[:, self.idx, :, :] # (N, d, H, W)
        
        print("Calculating Mean and Covariance...")
        # Calculate Mean and Covariance at each (h, w) location
        # Reshape to (H*W, N, d) for easier calculation or keep (H, W, N, d)
        
        self.mean = torch.mean(subset_embeddings, dim=0) # (d, H, W)
        self.cov = torch.zeros(C if self.idx is None else len(self.idx), C if self.idx is None else len(self.idx), H, W)
        
        # Compute Covariance for each spatial position
        # Efficient implementation: (N, d, H, W)
        # We need Cov matrix (d, d) for each (H, W)
        # This can be slow if done loop-wise.
        
        # Vectorized covariance calculation
        # Flatten spatial: (N, d, H*W) -> permute -> (H*W, d, N)
        B, d, H, W = subset_embeddings.shape
        flat = subset_embeddings.view(B, d, -1).permute(2, 1, 0) # (Pixels, d, N)
        
        # Mean per pixel: (Pixels, d, 1)
        mu = torch.mean(flat, dim=2, keepdim=True) 
        
        # Center data
        X = flat - mu
        
        # Cov = (1/(N-1)) * X * X.T
        # (Pixels, d, N) @ (Pixels, N, d) -> (Pixels, d, d)
        # Add regularization identity to avoid singular
        epsilon = 0.01
        
        print(f"Computing covariance for {H*W} pixels...")
        self.cov = torch.zeros(H*W, d, d)
        identity = torch.eye(d).unsqueeze(0).repeat(H*W, 1, 1)
        
        self.cov = (torch.bmm(X, X.permute(0, 2, 1)) / (B - 1)) + epsilon * identity
        
        # Store params
        self.mean = self.mean # (d, H, W)
        self.cov_inv = torch.inverse(self.cov).permute(1, 2, 0).view(d, d, H, W) # Pre-calculate inverse
        # Save indices
        self.idx = self.idx
        
        print("Training complete.")

    def save(self, path):
         with open(path, 'wb') as f:
            pickle.dump({
                'mean': self.mean, 
                'cov_inv': self.cov_inv, 
                'idx': self.idx,
                'backbone_name': 'resnet50'
            }, f)

def train(args):
    dataset = ChainDataset(root_dir=args.data_path, transform=None)
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset.transform = transform
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = PaDiM(backbone_name="resnet50", d_reduced=100)
    model.fit(dataloader)
    
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/train/good")
    parser.add_argument("--model_path", type=str, default="padim_resnet50.pkl")
    args = parser.parse_args()
    train(args)
