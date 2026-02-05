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
from src.segmentation import segment_jewelry, apply_mask_to_anomaly_map
from src.utils import plot_anomaly

class PatchCorePredictor:
    def __init__(self, model_path, backbone_name="resnet50"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load backbone
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.layers_to_extract = ['layer2', 'layer3']
        
        self.backbone.to(self.device)
        self.backbone.eval()
        
        self.features = []
        self._register_hooks()
        
        # Load memory bank
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.memory_bank = data['memory_bank']
            self.nbrs = data['nbrs']

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
        emb = self._embed(image_tensor)
        B, D, H_prime, W_prime = emb.shape
        emb_flat = emb.permute(0, 2, 3, 1).reshape(-1, D).cpu().numpy()
        
        distances, _ = self.nbrs.kneighbors(emb_flat)
        anomaly_map = distances.reshape(H_prime, W_prime)
        
        # Upsample anomaly map to original image resolution (done in visualization/post-proc usually, but here for score)
        # We usually do bilinear resize of anomaly map
        return anomaly_map

def predict(args):
    # 1. Load Image
    image = Image.open(args.image_path).convert('RGB')
    original_size = image.size
    
    # 2. Segment if requested
    mask = None
    if args.mask_background:
        print("Applying background masking...")
        _, mask = segment_jewelry(image)
        # Verify mask is not empty
        if mask.sum() == 0:
            print("Warning: Segmentation mask is empty! Falling back to full image.")
            mask = np.ones((image.height, image.width), dtype=np.float32)
        else:
             print("Segmentation successful.")
    
    # 3. Transform
    target_size = (256, 256)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    # 4. Model Prediction
    predictor = PatchCorePredictor(args.model_path)
    anomaly_map = predictor.predict(input_tensor)
    
    # 5. Post-process Anomaly Map
    # Resize to match original image size for accurate masking
    anomaly_map_resized = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)
    
    # 6. Apply Mask to Score Logic
    if args.mask_background and mask is not None:
        # Ensure mask matches anomaly map size
        if mask.shape != anomaly_map_resized.shape:
             mask = cv2.resize(mask, (anomaly_map_resized.shape[1], anomaly_map_resized.shape[0]))
             
        masked_map = anomaly_map_resized * mask
        # Score is max of foreground only
        # Avoid zero elements (background) dragging down average or min, but max is safe if background is 0
        # However, if background was high anomaly, zeroing it correctly removes it.
        # Check if foreground exists
        if mask.sum() > 0:
            final_score = np.max(masked_map)
        else:
            final_score = 0
    else:
        final_score = np.max(anomaly_map_resized)
        
    print(f"Anomaly Score: {final_score:.4f}")
    print(f"Prediction: {'NOT OK' if final_score > args.threshold else 'OK'}")
    
    if args.vis:
        # Re-transform for vis (tensor wrapper)
        vis_transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
        vis_tensor = vis_transform(image)
        
        base_name = os.path.basename(args.image_path)
        out_name = f"result_{os.path.splitext(base_name)[0]}_{'masked' if args.mask_background else 'baseline'}.png"
        
        plot_anomaly(vis_tensor, anomaly_map, final_score, args.threshold, out_name, mask if args.mask_background else None)
        print(f"Saved visualization to {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="patchcore_resnet50.pkl")
    parser.add_argument("--mask_background", action="store_true", help="Apply segmentation mask to anomaly map")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--threshold", type=float, default=7.55) # Use refined threshold
    args = parser.parse_args()
    predict(args)
