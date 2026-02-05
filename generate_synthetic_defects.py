import cv2
import numpy as np
import os
import glob
import random
from src.segmentation import segment_jewelry

def create_cut_defect(image, mask):
    """Simulates a cut/break by drawing a line through the chain."""
    img = image.copy()
    H, W = img.shape[:2]
    
    # improved logic: Find chain region from mask
    y_idxs, x_idxs = np.where(mask > 0)
    if len(y_idxs) == 0: return img
    
    # Pick a random point on the chain
    idx = random.randint(0, len(y_idxs)-1)
    cx, cy = x_idxs[idx], y_idxs[idx]
    
    # Draw a line (black/background color)
    angle = random.uniform(0, 360)
    length = random.randint(10, 30)
    thickness = random.randint(2, 5)
    
    x2 = int(cx + length * np.cos(np.radians(angle)))
    y2 = int(cy + length * np.sin(np.radians(angle)))
    x1 = int(cx - length * np.cos(np.radians(angle)))
    y1 = int(cy - length * np.sin(np.radians(angle)))
    
    # Color: Average background color or just black
    # Sample background color
    bg_mask = (mask == 0)
    if bg_mask.sum() > 0:
        bg_color = img[bg_mask].mean(axis=0).astype(int).tolist()
    else:
        bg_color = (0, 0, 0)
        
    cv2.line(img, (x1, y1), (x2, y2), bg_color, thickness)
    return img

def create_dent_defect(image, mask):
    """Simulates a dent by warping a small region."""
    # Simplified: Draw a dark spot or small distortion
    img = image.copy()
    y_idxs, x_idxs = np.where(mask > 0)
    if len(y_idxs) == 0: return img
    
    idx = random.randint(0, len(y_idxs)-1)
    cx, cy = x_idxs[idx], y_idxs[idx]
    
    radius = random.randint(5, 15)
    
    # Darken region to simulate dent shadow
    # Create circular mask
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    dist = np.sqrt((X - cx)**2 + (Y-cy)**2)
    dent_mask = dist <= radius
    
    img[dent_mask] = (img[dent_mask] * 0.6).astype(np.uint8) # Darken
    return img

def create_color_defect(image, mask):
    """Simulates discoloration (copper/rust spot)."""
    img = image.copy()
    y_idxs, x_idxs = np.where(mask > 0)
    if len(y_idxs) == 0: return img
    
    idx = random.randint(0, len(y_idxs)-1)
    cx, cy = x_idxs[idx], y_idxs[idx]
    
    radius = random.randint(10, 25)
    
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    dist = np.sqrt((X - cx)**2 + (Y-cy)**2)
    spot_mask = dist <= radius
    
    # Add reddish/copper tint
    # BGR -> increase R, decrease B
    if spot_mask.sum() > 0:
        region = img[spot_mask].astype(float)
        region[:, 2] *= 1.3 # R
        region[:, 0] *= 0.7 # B
        np.clip(region, 0, 255, out=region)
        img[spot_mask] = region.astype(np.uint8)
        
    return img

def create_joint_defect(image, mask):
    """Simulates an open joint (thin tiny gap)."""
    img = image.copy()
    y_idxs, x_idxs = np.where(mask > 0)
    if len(y_idxs) == 0: return img
    
    idx = random.randint(0, len(y_idxs)-1)
    cx, cy = x_idxs[idx], y_idxs[idx]
    
    # Very thin short line
    angle = random.uniform(0, 360)
    length = random.randint(5, 10)
    thickness = 1
    
    x2 = int(cx + length * np.cos(np.radians(angle)))
    y2 = int(cy + length * np.sin(np.radians(angle)))
    
    cv2.line(img, (cx, cy), (x2, y2), (0,0,0), thickness)
    return img

def generate_dataset():
    source_dir = "dataset/train/good"
    output_dir = "dataset/test/synthetic"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    defect_types = {
        "cut": create_cut_defect,
        "dent": create_dent_defect,
        "color": create_color_defect,
        "joint": create_joint_defect
    }
    
    image_paths = glob.glob(os.path.join(source_dir, "*.png")) + glob.glob(os.path.join(source_dir, "*.jpg"))
    # Use 10-20 images base
    image_paths = image_paths[:20]
    
    count = 0
    print("Generating synthetic defects...")
    
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            pil_img = cv2.imread(img_path) # BGR
            if pil_img is None: continue
            
            # Get Mask for placement
            # segment_jewelry expects PIL RGB
            # Convert for segmentation
            pil_rgb = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_obj = Image.fromarray(pil_rgb)
            _, mask_obj = segment_jewelry(pil_obj)
            mask = mask_obj # numpy array
            
            # Generate one of each defect type per image
            for d_name, d_func in defect_types.items():
                syn_img = d_func(pil_img, mask)
                full_name = f"{base_name}_{d_name}.png"
                cv2.imwrite(os.path.join(output_dir, full_name), syn_img)
                count += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    print(f"Generated {count} synthetic test images in {output_dir}")

if __name__ == "__main__":
    generate_dataset()
