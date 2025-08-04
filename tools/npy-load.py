import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_channels(input_dir, output_dir):
    """
    Load .npy files from input directory, visualize each channel, and save as PNG files.
    
    Args:
        input_dir (str): Path to directory containing .npy files
        output_dir (str): Path to directory where PNG files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Different colormaps for each channel
    colormaps = ['viridis', 'magma', 'plasma', 'inferno']
    
    # Process each .npy file in the input directory
    for npy_file in Path(input_dir).glob('*.npy'):
        # Load the data
        data = np.load(npy_file)
        
        # Verify the shape
        if data.shape != (4, 64, 64):
            print(f"Skipping {npy_file.name} - unexpected shape: {data.shape}")
            continue
            
        # Base filename without extension
        base_name = npy_file.stem
        
        # Create and save visualization for each channel
        for channel in range(4):
            # Create figure with no white space around the edges
            plt.figure(figsize=(6, 6))
            plt.margins(0, 0)
            
            # Plot the channel with specific colormap
            plt.imshow(data[channel], cmap=colormaps[channel])
            
            # Remove axes for cleaner look
            plt.axis('off')
            
            # Add colorbar
            # plt.colorbar(orientation='vertical', pad=0.05)
            
            # Create output filename
            output_filename = f"{base_name}_channel{channel+1}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save with tight layout and high DPI for quality
            plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0)
            plt.close()
            
            print(f"Saved {output_filename}")


from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
import matplotlib.pyplot as plt

def estimate_depth(image_path):
    """
    Estimate depth from a local image file using Depth Anything V2
    
    Args:
        image_path (str): Path to the input image file
    """
    # Load local image
    image = Image.open(image_path)
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-v2-base")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-v2-base")

    # Prepare image
    inputs = processor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Post-process the depth map
    depth_map = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Convert to numpy array and normalize for visualization
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_map = depth_map.astype(np.uint8)

    # Visualize results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title('Predicted Depth Map')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the depth map
    output_path = image_path.rsplit('.', 1)[0] + '_depth.png'
    depth_image = Image.fromarray(depth_map)
    depth_image.save(output_path)
    print(f"Depth map saved as: {output_path}")

    return depth_map

def estimate_depth_old(image_path):
    """
    Estimate depth from a local image file using DPT (Dense Prediction Transformer)
    
    Args:
        image_path (str): Path to the input image file
    """
    # Load local image
    image = Image.open(image_path)
    
    # Load model and processor
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # Prepare image
    inputs = processor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Post-process the depth map
    depth_map = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    # colormaps = ['viridis']

    # Convert to numpy array and normalize for visualization
    depth_map = depth_map.squeeze().cpu().numpy()
    
    # Normalize depth map
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_map = depth_map.astype(np.uint8)

    # Visualize results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Normalize the depth map to [0, 255] for visualization
    normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_colormap = cm.viridis(normalized_depth)  # Apply viridis colormap

    # Convert to an image (ensure it's in 8-bit format)
    depth_colormap_image = (depth_colormap[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel if present

    # Save the image
    output_path = image_path.rsplit('.', 1)[0] + '_depth_colormap.png'
    Image.fromarray(depth_colormap_image).save(output_path)

    # Save the depth map
    # output_path = image_path.rsplit('.', 1)[0] + '_depth.png'
    # depth_image = Image.fromarray(depth_map)
    # depth_image.save(output_path)
    # print(f"Depth map saved as: {output_path}")
    return depth_map

# # Example usage
# if __name__ == "__main__":
#     # Replace with your image path
#     basedir = "/home/student.unimelb.edu.au/xueyangk/Features"
#     filename = "47157250_8166731384.jpg"
#     # depth_map = estimate_depth(os.path.join(basedir, filename))
#     depth_map = estimate_depth_old(os.path.join(basedir, filename))

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual directories
    basedir = "/home/student.unimelb.edu.au/xueyangk" 
    input_folder = "pantheon_exterior/pantheon_exterior/feature"    #### "pantheon_exterior/pantheon_exterior/feature"
    #"Fusion-Features/tanks-and-temple-Scan2/feature"  ##### "Features/tanks-and-temple-Scan1/feature"   "Fusion-Features/tanks-and-temple-Scan3/feature"
    output_folder = "0128-output"

    input_directory = os.path.join(basedir, input_folder)
    output_directory = os.path.join(basedir, output_folder)

    visualize_channels(input_directory, output_directory)