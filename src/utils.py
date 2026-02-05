import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def normalize_anomaly_map(anomaly_map):
    """Normalizes anomaly map to 0-1 range."""
    min_val = anomaly_map.min()
    max_val = anomaly_map.max()
    if max_val - min_val > 0:
        return (anomaly_map - min_val) / (max_val - min_val)
    return anomaly_map

def plot_anomaly(image_tensor, anomaly_map, score, threshold, save_path, mask=None):
    """
    Plots the original image, anomaly map, and prediction result.
    """
    # Denormalize image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Resize anomaly map to image size
    if anomaly_map.shape != img_np.shape[:2]:
        anomaly_map = cv2.resize(anomaly_map, (img_np.shape[1], img_np.shape[0]))
        
    # Apply mask to anomaly map vis if provided
    if mask is not None:
         mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
         anomaly_map = anomaly_map * mask_resized

    norm_map = normalize_anomaly_map(anomaly_map)
    heatmap = cv2.applyColorMap(np.uint8(norm_map * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    overlay = 0.5 * img_np + 0.5 * heatmap
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    status = "OK" if score <= threshold else "NOT OK"
    color = "green" if status == "OK" else "red"
    axes[1].set_title(f"Anomaly Map | Score: {score:.4f} | {status}", color=color)
    axes[1].axis('off')
    
    # Mark max anomaly point
    if status == "NOT OK":
        y, x = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
        axes[1].plot(x, y, 'ro', markersize=5, markeredgewidth=1, markeredgecolor='white')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
