import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from train_cae import CAEn # Import model definition
from src.segmentation import segment_jewelry
from src.utils import plot_anomaly

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = CAEn().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Load Image
    image = Image.open(args.image_path).convert('RGB')
    original_size = image.size
    
    # Segment if requested
    mask = None
    if args.mask_background:
        print("Applying background masking...")
        _, mask = segment_jewelry(image)
        if mask.sum() == 0:
            print("Warning: Segmentation mask is empty!")
            mask = np.ones((image.height, image.width), dtype=np.float32)

    # Transform
    target_size = (256, 256)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        reconstruction = model(input_tensor)
        
    # Calculate Anomaly Map (Pixel-wise MSE)
    # diff shape: (1, 3, 256, 256)
    diff = (input_tensor - reconstruction) ** 2
    # Average across channels -> (1, 256, 256)
    anomaly_map = torch.mean(diff, dim=1).squeeze().cpu().numpy()
    
    # Post-process
    anomaly_map_resized = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Apply Mask logic
    if args.mask_background and mask is not None:
         # Resize mask to original if needed (though mask comes from original image size)
         if mask.shape != anomaly_map_resized.shape:
             mask = cv2.resize(mask, (anomaly_map_resized.shape[1], anomaly_map_resized.shape[0]))
             
         masked_map = anomaly_map_resized * mask
         if mask.sum() > 0:
             # Use 99th percentile to be robust against single pixel noise in AE
             final_score = np.percentile(masked_map[mask > 0], 99) 
             # Or max, but AE outputs can be noisy at edges
             # final_score = np.max(masked_map)
         else:
             final_score = 0
    else:
        # Without mask, background reconstruction error might be high if texture varies
        final_score = np.percentile(anomaly_map_resized, 99)
        
    print(f"Anomaly Score (MSE): {final_score:.6f}")
    # Threshold handling for AE is different, it's usually much lower, e.g., 0.01-0.05
    # We will need calibration for this too, but for predicting now we use user arg
    print(f"Prediction: {'NOT OK' if final_score > args.threshold else 'OK'}")
    
    if args.vis:
        # Re-transform for vis (already 0-1 tensor for plot_anomaly which expects normalized? 
        # Wait, utils.plot_anomaly expects normalized (mean/std subtraction).
        # CAE uses 0-1. We should adapt slightly.
        # Actually utils.plot_anomaly denormalizes using ImageNet mean/std. 
        # If we pass 0-1 tensor, it will look weird.
        # Let's create a "normalized" version just for plot_utils to be happy, or modify plot_utils?
        # Better: Feed utils.plot_anomaly a tensor that IT can denormalize. 
        # Since CAEn input is 0-1, (img - mean)/std will make it standard.
        
        vis_transform_std = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        vis_tensor = vis_transform_std(image)
        
        base_name = os.path.basename(args.image_path)
        out_name = f"result_{os.path.splitext(base_name)[0]}_CAE_{'masked' if args.mask_background else 'baseline'}.png"
        
        plot_anomaly(vis_tensor, anomaly_map, final_score, args.threshold, out_name, mask if args.mask_background else None)
        print(f"Saved visualization to {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="cae_model.pth")
    parser.add_argument("--mask_background", action="store_true", help="Apply segmentation mask")
    parser.add_argument("--vis", action="store_true")
    # Default threshold for MSE is usually small. User needs to calibrate.
    parser.add_argument("--threshold", type=float, default=0.02) 
    args = parser.parse_args()
    predict(args)
