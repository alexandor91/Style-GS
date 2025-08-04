import cv2
import numpy as np
from skimage import morphology  # Optional for post-processing
import os

def create_luminance_masks(image_path, dark_percentile=22, bright_percentile=90, scale=20):
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to luminance (normalized to [0, 1])
    luminance = np.dot(image_rgb, [0.310, 0.587, 0.114]) / 255.0
    
    # Calculate thresholds using percentiles
    dark_thresh = np.percentile(luminance, dark_percentile)
    bright_thresh = np.percentile(luminance, bright_percentile)
    
    # Create soft masks using sigmoid
    dark_mask = 1 / (1 + np.exp((luminance - dark_thresh) * scale))
    bright_mask = 1 / (1 + np.exp(-(luminance - bright_thresh) * scale))
    
    # Create binary masks
    shadow_mask = (dark_mask > 0.5).astype(np.uint8) * 255
    reflection_mask = (bright_mask > 0.5).astype(np.uint8) * 255
    
    # Post-processing (optional)
    def clean_mask(mask):
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=50)
        mask = morphology.closing(mask, morphology.disk(3))
        return mask.astype(np.uint8) * 255
    
    shadow_mask = clean_mask(shadow_mask)
    reflection_mask = clean_mask(reflection_mask)
    
    return shadow_mask, reflection_mask

base_dir = '/home/student.unimelb.edu.au/xueyangk/scan114/image'
filename = '000017.png'
# Usage
shadow_mask, reflection_mask = create_luminance_masks(os.path.join(base_dir, filename))

# Save results
cv2.imwrite("shadow_mask.png", shadow_mask)
cv2.imwrite("reflection_mask.png", reflection_mask)