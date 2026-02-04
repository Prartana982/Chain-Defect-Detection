import cv2
import numpy as np
from PIL import Image

def segment_jewelry(image):
    """
    Segments jewelry from the background using multiple techniques.
    Returns the masked image with background removed (set to black).
    
    Args:
        image: PIL Image in RGB format
        
    Returns:
        masked_image: PIL Image with background removed
        mask: Binary mask (numpy array) where 1=jewelry, 0=background
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Method 1: Adaptive thresholding (works well for metallic jewelry)
    # Jewelry is typically brighter/more reflective than background
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Edge-based segmentation
    edges = cv2.Canny(blur, 50, 150)
    
    # Dilate edges to create connected regions
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask from the largest contour (assuming jewelry is the main object)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Optional: If the largest contour is too small, use multiple contours
        total_area = gray.shape[0] * gray.shape[1]
        largest_area = cv2.contourArea(largest_contour)
        
        if largest_area < 0.05 * total_area:
            # Fallback: use threshold-based mask
            mask = thresh1
    else:
        # Fallback: use threshold-based mask
        mask = thresh1
    
    # Morphological operations to clean up the mask
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Apply Gaussian blur to smooth mask edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply mask to original image
    masked_array = img_array.copy()
    for c in range(3):  # Apply to all RGB channels
        masked_array[:, :, c] = cv2.bitwise_and(img_array[:, :, c], mask)
    
    # Convert back to PIL Image
    masked_image = Image.fromarray(masked_array)
    
    # Normalize mask to 0-1 for easier use
    mask_normalized = (mask / 255.0).astype(np.float32)
    
    return masked_image, mask_normalized


def apply_mask_to_anomaly_map(anomaly_map, mask):
    """
    Applies a mask to the anomaly map, setting background regions to zero.
    
    Args:
        anomaly_map: numpy array (H, W) containing anomaly scores
        mask: numpy array (H, W) with values 0-1, where 1=jewelry, 0=background
        
    Returns:
        masked_anomaly_map: numpy array with background regions zeroed out
    """
    # Resize mask to match anomaly map if needed
    if anomaly_map.shape != mask.shape:
        mask_resized = cv2.resize(mask, (anomaly_map.shape[1], anomaly_map.shape[0]))
    else:
        mask_resized = mask
    
    # Apply mask (element-wise multiplication)
    masked_anomaly_map = anomaly_map * mask_resized
    
    return masked_anomaly_map


def get_jewelry_bounding_box(mask):
    """
    Gets the bounding box of the jewelry region.
    
    Args:
        mask: Binary mask (H, W) with 0-1 values
        
    Returns:
        (x, y, w, h): Bounding box coordinates and dimensions
    """
    # Convert to uint8 for contour detection
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    else:
        # Return full image if no contours found
        return 0, 0, mask.shape[1], mask.shape[0]
