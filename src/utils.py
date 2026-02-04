import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)

def plot_anomaly(image, anomaly_map, score, threshold=None, save_path=None, mask=None):
    """
    Flags the anomaly point on the original image.
    
    Args:
        mask: Optional binary mask (H, W) to show segmented jewelry region
    """
    img = denormalize(image)
    img = (img * 255).astype(np.uint8)
    
    # Smooth the anomaly map to reduce noise and better localize defects
    anomaly_map = cv2.GaussianBlur(anomaly_map, (15, 15), 0)

    # Resize anomaly map to image size
    anomaly_map = cv2.resize(anomaly_map, (img.shape[1], img.shape[0]))
    
    # Find the point with maximum anomaly
    max_idx = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
    max_y, max_x = max_idx
    
    # Draw circle at anomaly point
    result_img = img.copy()
    cv2.circle(result_img, (max_x, max_y), radius=10, color=(0, 0, 255), thickness=2)  # Red circle
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    status = "NOT OK" if threshold and score > threshold else "OK" if threshold else "Score: {:.4f}".format(score)
    plt.title(f"Anomaly Point Flagged | {status}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
