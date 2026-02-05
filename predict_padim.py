import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import cv2
import os
from scipy.spatial.distance import mahalanobis
from src.segmentation import segment_jewelry, apply_mask_to_anomaly_map
from src.utils import plot_anomaly

class PaDiMPredictor:
    def __init__(self, model_path, backbone_name="resnet50"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load backbone
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.layers_to_extract = ['layer1', 'layer2', 'layer3']
        
        self.backbone.to(self.device)
        self.backbone.eval()
        
        self.features = []
        self._register_hooks()
        
        # Load params
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.mean = data['mean'].to(self.device) # (d, H, W)
            self.cov_inv = data['cov_inv'].to(self.device) # (d, d, H, W)
            self.idx = data['idx'].to(self.device)

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
        return torch.cat(embeddings, dim=1)

    def predict(self, image_tensor):
        # (1, C, H, W) -> (1, d, H_feat, W_feat)
        emb = self._embed(image_tensor)
        
        # Select indices
        emb = emb[:, self.idx, :, :]
        
        # Calculate Mahalanobis distance per pixel
        # Distance = sqrt((x - u).T * S^-1 * (x - u))
        # Vectorized implementation on GPU
        
        B, C, H, W = emb.shape
        # Flatten spatial: (1, C, H*W)
        x = emb.view(B, C, -1) 
        # Mean: (C, H*W)
        mu = self.mean.view(C, -1)
        # CovInv: (C, C, H*W)
        inv = self.cov_inv.view(C, C, -1)
        
        # Delta = x - mu
        delta = x - mu.unsqueeze(0) # (1, C, H*W)
        
        # We need (delta.T * inv * delta) per pixel
        # delta shape: (1, C, N) -> permute -> (N, 1, C)
        # inv shape: (C, C, N) -> permute -> (N, C, C)
        
        N = H * W
        delta_perm = delta.permute(2, 0, 1) # (N, 1, C)
        inv_perm = inv.permute(2, 0, 1) # (N, C, C)
        
        # (N, 1, C) @ (N, C, C) -> (N, 1, C)
        left_term = torch.bmm(delta_perm, inv_perm)
        # (N, 1, C) @ (N, C, 1) -> (N, 1, 1)
        dist_sq = torch.bmm(left_term, delta_perm.permute(0, 2, 1))
        
        dist = torch.sqrt(dist_sq).view(H, W)
        
        return dist.cpu().numpy()

def predict(args):
    # 1. Load Image
    image = Image.open(args.image_path).convert('RGB')
    original_size = image.size
    
    # 2. Segment
    mask = None
    if args.mask_background:
        print("Applying background masking...")
        _, mask = segment_jewelry(image)
        if mask.sum() == 0:
            print("Warning: Segmentation mask is empty!")
            mask = np.ones((image.height, image.width), dtype=np.float32)

    # 3. Transform
    target_size = (256, 256)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    # 4. Predict
    predictor = PaDiMPredictor(args.model_path)
    anomaly_map = predictor.predict(input_tensor)
    
    # 5. Post-process
    # Usually PaDiM output is quite smooth, but we resize to original
    anomaly_map_resized = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)
    
    # 6. Apply Mask
    if args.mask_background and mask is not None:
         if mask.shape != anomaly_map_resized.shape:
             mask = cv2.resize(mask, (anomaly_map_resized.shape[1], anomaly_map_resized.shape[0]))
             
         masked_map = anomaly_map_resized * mask
         if mask.sum() > 0:
            final_score = np.max(masked_map)
         else:
            final_score = 0
    else:
        final_score = np.max(anomaly_map_resized)
        
    print(f"Anomaly Score: {final_score:.4f}")
    # PaDiM distances are Mahalanobis, typically larger range than PatchCore or CAE
    print(f"Prediction: {'NOT OK' if final_score > args.threshold else 'OK'}")
    
    if args.vis:
        vis_transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
        vis_tensor = vis_transform(image)
        
        base_name = os.path.basename(args.image_path)
        out_name = f"result_{os.path.splitext(base_name)[0]}_PaDiM_{'masked' if args.mask_background else 'baseline'}.png"
        
        plot_anomaly(vis_tensor, anomaly_map, final_score, args.threshold, out_name, mask if args.mask_background else None)
        print(f"Saved visualization to {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="padim_resnet50.pkl")
    parser.add_argument("--mask_background", action="store_true", help="Apply segmentation mask")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--threshold", type=float, default=15.0) # Placeholder threshold for PaDiM
    args = parser.parse_args()
    predict(args)
