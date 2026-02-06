import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import pickle
import os
from src.dataset import ChainDataset

class PatchCore(nn.Module):
    def __init__(self, backbone_name="resnet50", memory_bank_size=1000):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_bank_size = memory_bank_size
        
        # Load backbone
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.layers_to_extract = ['layer2', 'layer3']
        else:
            raise NotImplementedError("Only ResNet50 supported for now")
        
        self.backbone.to(self.device)
        self.backbone.eval()
        
        self.features = []
        self._register_hooks()
        
        self.memory_bank = None
        self.nbrs = None

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
        
        ref_h, ref_w = self.features[0].shape[2], self.features[0].shape[3]
        
        embeddings = []
        for feat in self.features:
            feat = F.interpolate(feat, size=(ref_h, ref_w), mode='bilinear', align_corners=True)
            embeddings.append(feat)
            
        embedding = torch.cat(embeddings, dim=1) 
        return embedding

    def fit(self, dataloader):
        print("Extracting features from training data...")
        embedding_list = []
        
        for images, _, _ in tqdm(dataloader):
            emb = self._embed(images)
            # shape: (B, D, H', W') -> (B, H', W', D)
            emb = emb.permute(0, 2, 3, 1).contiguous()
            emb = emb.reshape(-1, emb.shape[-1]) 
            embedding_list.append(emb.cpu())
            
        full_embeddings = torch.cat(embedding_list, dim=0)
        
        print(f"Total patches: {full_embeddings.shape[0]}. Subsampling to {self.memory_bank_size}...")
        if full_embeddings.shape[0] > self.memory_bank_size:
            indices = np.random.choice(full_embeddings.shape[0], self.memory_bank_size, replace=False)
            self.memory_bank = full_embeddings[indices]
        else:
            self.memory_bank = full_embeddings
            
        print("Building Nearest Neighbors index...")
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=2)
        self.nbrs.fit(self.memory_bank.numpy())
        print("Training complete.")
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'memory_bank': self.memory_bank, 'nbrs': self.nbrs}, f)

def train(args):
    dataset = ChainDataset(root_dir=args.data_path, transform=None) # Transforms handled in embedding? No, need tensor conversion
    
    # Define transforms
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Simple resize for PoC
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset.transform = transform
    
    if len(dataset) == 0:
        print(f"No images found in {args.data_path}")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = PatchCore(backbone_name="resnet50", memory_bank_size=args.bank_size)
    model.fit(dataloader)
    
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/train/good")
    parser.add_argument("--model_path", type=str, default="patchcore_resnet50.pkl")
    parser.add_argument("--bank_size", type=int, default=1000)
    args = parser.parse_args()
    train(args)
