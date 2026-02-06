import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from src.dataset import ChainDataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DINOv2 (ViT-S/14) from TorchHub - powerful generic features
    # Or DINO v1
    print("Loading DINOv2 model...")
    # Using dinov2_vits14 for good trade-off
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    backbone.to(device)
    backbone.eval()
    
    dataset = ChainDataset(root_dir=args.data_path)
    # DINOv2 expects multiple of 14, standard resize 224 or 252 (14*18)
    transform = transforms.Compose([
        transforms.Resize((252, 252)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset.transform = transform
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    features_list = []
    
    print("Extracting features from training data...")
    with torch.no_grad():
        for images, _, _ in tqdm(dataloader):
            images = images.to(device)
            # DINOv2 forward_features returns dict including 'x_norm_patchtokens' and 'x_norm_clstoken'
            # We use patch tokens for localized anomaly detection
            ret = backbone.forward_features(images)
            patch_tokens = ret['x_norm_patchtokens'] # (B, N_patches, D)
            
            # Reshape patches to spatial if needed or just keep as list of vectors
            # (B, H*W, D) -> (B*H*W, D)
            B, N, D = patch_tokens.shape
            features_list.append(patch_tokens.reshape(-1, D).cpu().numpy())
            
    full_features = np.concatenate(features_list, axis=0)
    
    # Subsample for memory bank (DINO dense features can be large)
    print(f"Total features: {full_features.shape[0]}. Subsampling to {args.bank_size}...")
    if full_features.shape[0] > args.bank_size:
        indices = np.random.choice(full_features.shape[0], args.bank_size, replace=False)
        memory_bank = full_features[indices]
    else:
        memory_bank = full_features
        
    print("Building Nearest Neighbors index...")
    # DINO features are cosine-similarity friendly, but Euclidean usually works fine on normalized features.
    # DINOv2 output is already LayerNormed.
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=2)
    nbrs.fit(memory_bank)
    
    with open(args.model_path, 'wb') as f:
        pickle.dump({'memory_bank': memory_bank, 'nbrs': nbrs}, f)
        
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/train/good")
    parser.add_argument("--model_path", type=str, default="dino_vits14.pkl")
    parser.add_argument("--bank_size", type=int, default=5000)
    args = parser.parse_args()
    train(args)
