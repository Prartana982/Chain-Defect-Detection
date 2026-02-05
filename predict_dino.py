import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import cv2
import os
from src.segmentation import segment_jewelry, apply_mask_to_anomaly_map
from src.utils import plot_anomaly

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading DINOv2 model...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    backbone.to(device)
    backbone.eval()
    
    # Load memory bank
    with open(args.model_path, 'rb') as f:
        data = pickle.load(f)
        nbrs = data['nbrs']
        
    # Load Image
    image = Image.open(args.image_path).convert('RGB')
    original_size = image.size
    
    # Segment
    mask = None
    if args.mask_background:
        _, mask = segment_jewelry(image)
        if mask.sum() == 0:
            mask = np.ones((image.height, image.width), dtype=np.float32)

    # Transform
    # 252 is 14*18, closest multiple of 14 to 256
    target_size = (252, 252) 
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ret = backbone.forward_features(input_tensor)
        patch_tokens = ret['x_norm_patchtokens'] # (1, N, D)
        
    # Spatial shape
    # With 252x252 input and patch size 14, output is 18x18
    H_feat, W_feat = target_size[0] // 14, target_size[1] // 14
    
    features = patch_tokens[0].cpu().numpy() # (N, D)
    
    # NN Search
    distances, _ = nbrs.kneighbors(features) # (N, 1)
    
    # Reshape to map
    anomaly_map = distances.reshape(H_feat, W_feat)
    
    # Post-process
    anomaly_map_resized = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)
    
    if args.mask_background and mask is not None:
         if mask.shape != anomaly_map_resized.shape:
             mask = cv2.resize(mask, (anomaly_map_resized.shape[1], anomaly_map_resized.shape[0]))
         masked_map = anomaly_map_resized * mask
         final_score = np.max(masked_map) if mask.sum() > 0 else 0
    else:
        final_score = np.max(anomaly_map_resized)
        
    print(f"Anomaly Score: {final_score:.4f}")
    print(f"Prediction: {'NOT OK' if final_score > args.threshold else 'OK'}")
    
    if args.vis:
        vis_transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
        vis_tensor = vis_transform(image)
        base_name = os.path.basename(args.image_path)
        out_name = f"result_{os.path.splitext(base_name)[0]}_DINO_{'masked' if args.mask_background else 'baseline'}.png"
        plot_anomaly(vis_tensor, anomaly_map, final_score, args.threshold, out_name, mask if args.mask_background else None)
        print(f"Saved visualization to {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="dino_vits14.pkl")
    parser.add_argument("--mask_background", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()
    predict(args)
