import argparse
import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils.data import DataLoader
from src.dataset import ChainDataset
from src.model import PatchCore
from src.utils import plot_anomaly
from src.segmentation import segment_jewelry, apply_mask_to_anomaly_map
from PIL import Image
from torchvision import transforms

def train(args):
    dataset = ChainDataset(root_dir=args.data_path, segment_jewelry=args.segment_jewelry)
    if len(dataset) == 0:
        print(f"No images found in {args.data_path}. Please add images to dataset/train/good/")
        return

    if args.segment_jewelry:
        print("Training with jewelry segmentation enabled...")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = PatchCore(backbone_name="resnet50", memory_bank_size=args.bank_size)
    model.fit(dataloader)
    
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

def predict(args):
    model = PatchCore(backbone_name="resnet50")
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}. Please train first.")
        return
    model.load(args.model_path)
    
    # Sliding Window Prediction Logic
    image = Image.open(args.image_path).convert('RGB')
    
    # NEW: Segment jewelry from background if enabled
    original_image = image.copy()
    jewelry_mask = None
    
    if args.segment_jewelry:
        print("Segmenting jewelry from background...")
        image, jewelry_mask = segment_jewelry(image)
        print("Segmentation complete. Analyzing jewelry only.")
    
    # 1. Resize to height 224, maintaining aspect ratio
    target_height = 224
    aspect_ratio = image.width / image.height
    target_width = int(target_height * aspect_ratio)
    image = image.resize((target_width, target_height))
    
    # Resize mask to match resized image
    if jewelry_mask is not None:
        jewelry_mask = cv2.resize(jewelry_mask, (target_width, target_height))
    
    # 2. Transform for patches (No additional resize/crop)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_tensor = transform(image) # (C, H, W)
    
    # 3. Slide 224x224 window
    patch_size = 224
    stride = 112 # 50% overlap
    
    full_anomaly_map = np.zeros((target_height, target_width))
    count_map = np.zeros((target_height, target_width))
    
    x_coords = list(range(0, target_width - patch_size + 1, stride))
    if not x_coords or (target_width - patch_size < 0):
        # Image smaller than patch or single patch
        x_coords = [0]
    
    # Ensure coverage of the last part
    if x_coords[-1] + patch_size < target_width:
        x_coords.append(target_width - patch_size)
    
    # Prevent negative indexing if image is smaller than patch
    if target_width < patch_size:
        # Pad image or just resize to 224x224? 
        # For simplicity, if smaller, we already resized H to 224. 
        # If W < 224, we should probably pad. 
        # But let's assume chain > 224 width.
        x_coords = [0]
    
    print(f"Scanning {len(x_coords)} patches...")
    
    for x in x_coords:
        # Extract patch
        if target_width < patch_size:
             # Fallback for tiny images: pad width
             patch = torch.nn.functional.pad(full_tensor, (0, patch_size - target_width, 0, 0))
        else:
             patch = full_tensor[:, :, x:x+patch_size]
             
        input_tensor = patch.unsqueeze(0)
        
        _, amap = model.predict(input_tensor)
        
        # Resize amap (typically 28x28) to patch size (224x224)
        amap_resized = cv2.resize(amap, (patch_size, patch_size))
        
        # Stitch
        current_width = patch_size if target_width >= patch_size else target_width
        if target_width < patch_size:
             amap_resized = amap_resized[:, :target_width]
        
        full_anomaly_map[:, x:x+current_width] += amap_resized
        count_map[:, x:x+current_width] += 1
        
    # Average overlaps
    full_anomaly_map /= count_map
    
    # NEW: Apply mask to anomaly map if segmentation was used
    if jewelry_mask is not None:
        print("Applying jewelry mask to anomaly map...")
        full_anomaly_map = apply_mask_to_anomaly_map(full_anomaly_map, jewelry_mask)
        # Recalculate score only from jewelry regions (ignore background zeros)
        masked_values = full_anomaly_map[jewelry_mask > 0.5]
        if len(masked_values) > 0:
            final_score = np.max(masked_values)
        else:
            final_score = 0.0
    else:
        final_score = np.max(full_anomaly_map)
    
    # Final Score
    result = "NOT OK" if final_score > args.threshold else "OK"
    print(f"{{'anomaly_score': {final_score:.4f}, 'prediction': '{result}'}}")
    
    if args.vis:
        base_name = os.path.basename(args.image_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"result_{file_name_no_ext}_{timestamp}.png"
        plot_anomaly(full_tensor, full_anomaly_map, final_score, args.threshold, save_path=output_file, mask=jewelry_mask)
        print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Chain Defect Detection PoC")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    # Train Parser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data_path", type=str, default="dataset/train/good")
    train_parser.add_argument("--model_path", type=str, default="model.pkl")
    train_parser.add_argument("--bank_size", type=int, default=1000)
    train_parser.add_argument("--segment_jewelry", action="store_true", help="Segment jewelry from background during training")
    
    # Predict Parser
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--image_path", type=str, required=True)
    predict_parser.add_argument("--model_path", type=str, default="model.pkl")
    predict_parser.add_argument("--threshold", type=float, default=7.08) # Tuned based on calibration (Max Good 7.12)
    predict_parser.add_argument("--vis", action="store_true", help="Visualize output")
    predict_parser.add_argument("--segment_jewelry", action="store_true", help="Segment jewelry from background before analysis")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
