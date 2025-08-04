import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

base_dir = '/home/student.unimelb.edu.au/xueyangk/data'
filename = 'sh.npy'
point_filename = 'xyz.npy'
save_dir = os.path.join(base_dir, 'wigner_d_plots')

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Load SH coefficients
sh_coeffs = torch.from_numpy(np.load(os.path.join(base_dir, filename)))
points = torch.from_numpy(np.load(os.path.join(base_dir, point_filename)))

# Simulated Gaussian means for selection (Replace with actual data)
gaussian_means = torch.rand(points.shape[0], 3) * 2 - 1  # Shape: [N_gaussian, 3]

# Select SH coefficients based on a region
region_mask = (gaussian_means[:, 0] > -0.5) & (gaussian_means[:, 0] < 0.5) & \
              (gaussian_means[:, 1] > -0.5) & (gaussian_means[:, 1] < 0.5)
selected_sh = sh_coeffs[region_mask]  # Shape: [N_SH, 16, 3]

# Extract first-order (l=1) and second-order (l=2) SH coefficients separately
sh_first_order = selected_sh[:, 1:4, :]  # Shape: [N_SH, 3, 3]
sh_second_order = selected_sh[:, 4:9, :]  # Shape: [N_SH, 5, 3]

# Function to plot rotated SH coefficients with consistent color bar limits
def plot_rotated_sh(rotated_first_order, rotated_second_order, filename):
    vmin = min(rotated_first_order.min(), rotated_second_order.min())
    vmax = max(rotated_first_order.max(), rotated_second_order.max())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(3):  # Iterate over RGB channels
        im1 = axes[0, i].imshow(rotated_first_order[:, :, i], cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'First-order SH - Channel {i}')
        fig.colorbar(im1, ax=axes[0, i])
        
        im2 = axes[1, i].imshow(rotated_second_order[:, :, i], cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Second-order SH - Channel {i}')
        fig.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Iterate over rotations (0° to 360° with 5° steps)
for angle in range(0, 361, 5):
    angle_rad = np.radians(angle)
    
    # Compute Wigner-D matrices for l=1 (3×3) and l=2 (5×5)
    wigner_d_1 = torch.tensor(wigner_D_matrix(1, angle_rad, 0, 0), dtype=torch.float32)
    wigner_d_2 = torch.tensor(wigner_D_matrix(2, angle_rad, 0, 0), dtype=torch.float32)
    
    # Apply rotation to first and second-order SH coefficients separately
    rotated_first_order = torch.einsum('ij,bjk->bik', wigner_d_1, sh_first_order)
    rotated_second_order = torch.einsum('ij,bjk->bik', wigner_d_2, sh_second_order)
    
    # Save the rotated SH coefficients plot
    filename = os.path.join(save_dir, f'rotated_sh_{angle}.png')
    plot_rotated_sh(rotated_first_order, rotated_second_order, filename)