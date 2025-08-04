import os
import numpy as np
import scipy.signal
from PIL import Image
import torch
import lpips
import cv2
from scipy.spatial.distance import directed_hausdorff

def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    print(f'Initializing LPIPS: {net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    lpips_model = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return lpips_model(gt, im, normalize=True).item()

def rgb_ssim(img0, img1, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)

    mu0, mu1 = filt_fn(img0), filt_fn(img1)
    mu00, mu11, mu01 = mu0 * mu0, mu1 * mu1, mu0 * mu1
    sigma00, sigma11 = filt_fn(img0**2) - mu00, filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    sigma00, sigma11 = np.maximum(0., sigma00), np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))

    c1, c2 = (k1 * max_val)**2, (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    return np.mean(numer / denom)

def compute_l1_l2_loss(edges1, edges2):
    """
    Compute L1 and L2 loss between two 2D grayscale images.
    
    Args:
        edges1 (np.ndarray): First grayscale image (2D array).
        edges2 (np.ndarray): Second grayscale image (2D array).
    
    Returns:
        l1_loss (float): Mean absolute error (L1 loss).
        l2_loss (float): Mean squared error (L2 loss).
    """
    # Ensure both arrays are float for proper computation
    edges1 = edges1.astype(np.float32)
    edges2 = edges2.astype(np.float32)

    # Compute L1 loss (Mean Absolute Error)
    l1_loss = np.mean(np.abs(edges1 - edges2))

    # Compute L2 loss (Mean Squared Error)
    l2_loss = np.mean((edges1 - edges2) ** 2)

    return l1_loss, l2_loss


############## opencv canny edge detection #############
def extract_edges(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200) / 255.0
    return edges

def chamfer_distance(edges1_tensor, edges2_tensor):
    """
    Compute Chamfer Distance between two edge maps.
    """
    # Ensure input is a PyTorch tensor
    if isinstance(edges1_tensor, np.ndarray):
        edges1 = torch.from_numpy(edges1_tensor)
    if isinstance(edges2_tensor, np.ndarray):
        edges2 = torch.from_numpy(edges2_tensor)

    edges1 = edges1.to(torch.float32)
    edges2 = edges2.to(torch.float32)
    # # Invert the mask: 1 becomes 0, 0 becomes 1
    edges1 = 1 - edges1
    edges2 = 1 - edges2
    edges1[edges1 < 0.5] = 0
    edges2[edges2 < 0.5] = 0


    # Get nonzero (edge) points
    edges1_points = edges1.nonzero().float()  # (N1, 2)
    edges2_points = edges2.nonzero().float()  # (N2, 2)

    ######3 optional downsampling #############
    sample_rate = 1.0
    # **Subsample points to reduce memory usage**
    if sample_rate < 1.0:
        num_sample1 = max(1, int(edges1_points.shape[0] * sample_rate))
        num_sample2 = max(1, int(edges2_points.shape[0] * sample_rate))
        indices1 = torch.randperm(edges1_points.shape[0])[:num_sample1]
        indices2 = torch.randperm(edges2_points.shape[0])[:num_sample2]
        edges1_points = edges1_points[indices1]
        edges2_points = edges2_points[indices2]

    if edges1_points.shape[0] == 0 or edges2_points.shape[0] == 0:
        return torch.tensor(float('inf'))  # Handle empty edges case

    # Compute pairwise squared Euclidean distances
    dist1 = torch.cdist(edges1_points, edges2_points, p=2)  # (N1, N2)
    
    # Chamfer distance: average nearest-neighbor distances
    cd_1 = torch.mean(torch.min(dist1, dim=1)[0])  # Nearest from edges1 → edges2
    cd_2 = torch.mean(torch.min(dist1, dim=0)[0])  # Nearest from edges2 → edges1
    
    return cd_1 + cd_2  # Symmetric Chamfer distance


def hausdorff_distance(edges1, edges2):
    """
    Compute Hausdorff Distance between two edge maps.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(edges1, torch.Tensor):
        edges1 = edges1.cpu().numpy()
    if isinstance(edges2, torch.Tensor):
        edges2 = edges2.cpu().numpy()

    edges1 = 1 - edges1
    edges2 = 1 - edges2
    edges1[edges1 < 0.5] = 0
    edges2[edges2 < 0.5] = 0
    # Get nonzero (edge) points as coordinate lists
    edges1_points = np.array(np.nonzero(edges1)).T  # (N1, 2)
    edges2_points = np.array(np.nonzero(edges2)).T  # (N2, 2)

    if edges1_points.shape[0] == 0 or edges2_points.shape[0] == 0:
        return float('inf')  # Handle empty edges case
    sample_rate = 1.0
    # **Subsample points to reduce memory usage**
    if sample_rate < 1.0:
        num_sample1 = max(1, int(edges1_points.shape[0] * sample_rate))
        num_sample2 = max(1, int(edges2_points.shape[0] * sample_rate))
        indices1 = torch.randperm(edges1_points.shape[0])[:num_sample1]
        indices2 = torch.randperm(edges2_points.shape[0])[:num_sample2]
        edges1_points = edges1_points[indices1]
        edges2_points = edges2_points[indices2]

    if edges1_points.shape[0] == 0 or edges2_points.shape[0] == 0:
        return torch.tensor(float('inf'))  # Handle empty edges case

    # Compute directed Hausdorff distances
    hausdorff_1 = directed_hausdorff(edges1_points, edges2_points)[0]  # From edges1 to edges2
    hausdorff_2 = directed_hausdorff(edges2_points, edges1_points)[0]  # From edges2 to edges1

    return max(hausdorff_1, hausdorff_2)  # Symmetric Hausdorff distance

# def chamfer_distance(edges1, edges2):
#     cd = ChamferDistance()
#     edges1_tensor = torch.tensor(edges1, dtype=torch.float32).unsqueeze(0)
#     edges2_tensor = torch.tensor(edges2, dtype=torch.float32).unsqueeze(0)
#     return cd(edges1_tensor, edges2_tensor).item()

# def hausdorff_distance(edges1, edges2):
#     return max(directed_hausdorff(edges1, edges2)[0], directed_hausdorff(edges2, edges1)[0])

def get_sorted_files(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith('.png')])

def evaluate_metrics(gt_dir, generated_dir,  gt_view_edge_directory, generated_view_edge_directory, device='cuda'):
    gt_files = get_sorted_files(gt_dir)
    generated_files = get_sorted_files(generated_dir)
    gt_edge_files = get_sorted_files(gt_view_edge_directory)
    gen_edge_files = get_sorted_files(generated_view_edge_directory)
    paired_gt_files = []
    paired_generate_files = []
    paired_gt_edges = []
    paired_generated_edges = []
    ######### if generated views downsmapled less than gt views #######
    for generated_file in generated_files:
        for gt_file in gt_files:
            if generated_file == gt_file:         
                paired_gt_files.append(gt_file) 
                paired_generate_files.append(generated_file) 

    for gen_edge in gen_edge_files:
        for gt_edge in gt_edge_files:
            if gen_edge == gt_edge:         
                paired_gt_edges.append(gt_edge) 
                paired_generated_edges.append(gen_edge) 
    print(paired_gt_files)
    print(paired_generate_files)
    all_psnr, all_ssim, all_alex, all_vgg, all_rmse, all_cd, all_hd = [], [], [], [], [], [], []
    all_l1_error, all_l2_error = [], []
    for gt_file, gen_file, gt_edge_file, gen_edge_file in zip(paired_gt_files, paired_generate_files, paired_gt_edges, paired_generated_edges):
        gt_path, gen_path = os.path.join(gt_dir, gt_file), os.path.join(generated_dir, gen_file)
        gt_edge_path, gen_edge_path = os.path.join(gt_view_edge_directory, gt_edge_file), os.path.join(generated_view_edge_directory, gen_edge_file)
        
        gt = np.asarray(Image.open(gt_path), dtype=np.float32) / 255.0
        img = np.asarray(Image.open(gen_path), dtype=np.float32) / 255.0

        gt_edge = np.asarray(Image.open(gt_edge_path), dtype=np.float32) / 255.0
        img_edge = np.asarray(Image.open(gen_edge_path), dtype=np.float32) / 255.0
        
        # gtmask = gt[..., [3]] if gt.shape[-1] == 4 else np.ones_like(gt[..., :1])
        # gt = gt[..., :3] * gtmask + (1 - gtmask)

        psnr = -10. * np.log10(np.mean(np.square(img - gt)))
        ssim = rgb_ssim(img, gt, 1)
        lpips_alex = rgb_lpips(gt, img, 'alex', device)
        lpips_vgg = rgb_lpips(gt, img, 'vgg', device)
        rmse = np.sqrt(np.mean(np.square(img - gt)))
        l1_edge_error, l2_edge_error = compute_l1_l2_loss(gt, img)
        ########### canny edge detection ###############
        # gt_edge = extract_edges(gt)
        # img_edge = extract_edges(img)
        cd = chamfer_distance(gt_edge, img_edge)
        hd = hausdorff_distance(gt_edge, img_edge)
        print("########### metric results ###################")
        print(psnr)
        print(ssim)
        print(lpips_alex)
        print(lpips_vgg)
        print(rmse)
        print(cd)
        print(hd)
        print(l1_edge_error)
        print(l2_edge_error)

        all_l1_error.append(l1_edge_error)
        all_l2_error.append(l2_edge_error)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_alex.append(lpips_alex)
        all_vgg.append(lpips_vgg)
        all_rmse.append(rmse)
        all_cd.append(cd)
        all_hd.append(hd)
        # cd = None
        # hd = None

    print(f"Mean PSNR: {np.mean(all_psnr):.3f}, SSIM: {np.mean(all_ssim):.3f}, LPIPS-Alex: {np.mean(all_alex):.3f}, LPIPS-VGG: {np.mean(all_vgg):.3f}, RMSE: {np.mean(all_rmse):.3f}, Chamfer Dist: {np.mean(all_cd):.3f}, Hausdorff Dist: {np.mean(all_hd):.3f}")

print("@@@@@@@@@@@@@")
base_dir = "/home/student.unimelb.edu.au/xueyangk"
gt_folder = "gt_images/Family"
output_folder = "styled_images_SG/Family/0c2ea02018c720ea22625e892c31b82fc"   
#### Family_00c37783379d75b05384de1a96cd6e00.jpg

# Example usage
gt_edge_folder = "gt_edges/Family"
output_edge_folder = "styled_edges_SG/Family/0c2ea02018c720ea22625e892c31b82fc"   # Family_00c37783379d75b05384de1a96cd6e00.jpg
gt_directory = os.path.join(base_dir, gt_folder)
generated_directory = os.path.join(base_dir, output_folder)

gt_view_edge_directory = os.path.join(base_dir, gt_edge_folder)
generated_view_edge_directory = os.path.join(base_dir, output_edge_folder)

evaluate_metrics(gt_directory, generated_directory, gt_view_edge_directory, generated_view_edge_directory)