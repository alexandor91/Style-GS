import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gaussian_stylization_loss(rendered_img, style_img, content_img, D_origin, D_render, 
                            delta_scale, delta_opacity, F_render, F_style, F_content,
                            lambda_recolor=1.0, lambda_style=1.0, lambda_content=1.0,
                            lambda_depth=1.0, lambda_scale=0.1, lambda_opacity=0.1, 
                            lambda_tv=0.1):
    """
    Combined loss function for 3D Gaussian stylization including all components
    
    Args:
        rendered_img: Rendered/stylized image (B, C, H, W)
        style_img: Style reference image (B, C, H, W)
        content_img: Content reference image (B, C, H, W)
        D_origin: Original depth map (B, 1, H, W)
        D_render: Rendered depth map (B, 1, H, W)
        delta_scale: Scale changes (B, M)
        delta_opacity: Opacity changes (B, M)
        F_render: VGG features of rendered image
        F_style: VGG features of style image
        F_content: VGG features of content image
        lambda_*: Weight factors for different losses
        
    Returns:
        tuple: (total_loss, dict of individual losses)
    """
    # 0. recolor
    # 颜色空间变换 直接 A b 不带参数的匹配（）
    
    # 1. Recolor Loss (combining L1 and SSIM)
    def compute_recon_loss(I_content, I_render, window_size=11):
        # SSIM constants
        C1, C2 = 0.01**2, 0.03**2
        
        # Gaussian window
        sigma = 1.5
        gauss = torch.tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
                             for x in range(window_size)]).to(I_content.device)
        gauss_window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        gauss_window = gauss_window / gauss_window.sum()
        gauss_window = gauss_window.unsqueeze(0).unsqueeze(0)
        gauss_window = gauss_window.expand(I_content.size(1), 1, window_size, window_size)
        
        # Calculate SSIM
        pad = window_size // 2
        mu1 = F.conv2d(I_content, gauss_window, padding=pad, groups=I_content.shape[1])
        mu2 = F.conv2d(I_render, gauss_window, padding=pad, groups=I_render.shape[1])
        
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(I_content * I_content, gauss_window, padding=pad, groups=I_content.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(I_render * I_render, gauss_window, padding=pad, groups=I_render.shape[1]) - mu2_sq
        sigma12 = F.conv2d(I_content * I_render, gauss_window, padding=pad, groups=I_content.shape[1]) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_loss = 1 - ssim_map.mean()
        
        # L1 Loss
        l1_loss = torch.mean(torch.abs(I_content - I_render))
        
        return 0.8 * l1_loss + 0.2 * ssim_loss

    # 2. Style Feature Loss (nearest neighbor matching)
    def compute_style_loss(F_render, F_style):
        B, C, H, W = F_render.shape
        F_render_flat = F_render.view(B, C, -1)
        F_style_flat = F_style.view(B, C, -1)
        
        # Normalize features
        F_render_norm = F.normalize(F_render_flat, dim=1)
        F_style_norm = F.normalize(F_style_flat, dim=1)
        
        # Compute similarity matrix
        similarity = torch.bmm(F_render_norm.transpose(1, 2), F_style_norm)
        
        # Find nearest neighbors
        _, nn_indices = similarity.max(dim=2)
        
        # Gather matched features
        batch_indices = torch.arange(B, device=F_render.device).view(-1, 1).expand(-1, H*W)
        nn_features = F_style_flat[batch_indices, :, nn_indices]
        
        # Compute loss
        loss = 1 - torch.sum(F_render_norm * nn_features, dim=1).mean()
        
        return loss

    # 3. Content Loss
    def compute_content_loss(F_content, F_render):
        H, W = F_content.shape[2:]
        return torch.sum((F_content - F_render) ** 2) / (H * W)

    # 4. Depth Loss
    def compute_depth_loss(D_origin, D_render):
        H, W = D_origin.shape[2:]
        return torch.sum((D_origin - D_render) ** 2) / (H * W)

    # 5. Scale/Opacity Regularization
    def compute_reg_losses(delta_scale, delta_opacity):
        M = delta_scale.shape[1]
        scale_reg = torch.sum(torch.abs(delta_scale)) / M
        opacity_reg = torch.sum(torch.abs(delta_opacity)) / M
        return scale_reg, opacity_reg

    # 6. Total Variation Loss
    def compute_tv_loss(rendered_img):
        diff_h = rendered_img[..., 1:, :] - rendered_img[..., :-1, :]
        diff_v = rendered_img[..., :, 1:] - rendered_img[..., :, :-1]
        return (torch.sum(torch.abs(diff_h)) + torch.sum(torch.abs(diff_v))) / \
               (rendered_img.shape[2] * rendered_img.shape[3])

    # Calculate all losses
    recolor = compute_recolor_loss(content_img, rendered_img)
    style = compute_style_loss(F_render, F_style)
    content = compute_content_loss(F_content, F_render)
    depth = compute_depth_loss(D_origin, D_render)
    scale_reg, opacity_reg = compute_reg_losses(delta_scale, delta_opacity)
    tv = compute_tv_loss(rendered_img)

    # Combine all losses
    total_loss = (
        lambda_recolor * recolor +
        lambda_style * style +
        lambda_content * content +
        lambda_depth * depth +
        lambda_scale * scale_reg +
        lambda_opacity * opacity_reg +
        lambda_tv * tv
    )

    # Return total loss and components
    return total_loss, {
        'recolor_loss': recolor.item(),
        'style_loss': style.item(),
        'content_loss': content.item(),
        'depth_loss': depth.item(),
        'scale_reg': scale_reg.item(),
        'opacity_reg': opacity_reg.item(),
        'tv_loss': tv.item(),
        'total_loss': total_loss.item()
    }

# Example usage:
"""
# Example dimensions
B, C, H, W = 1, 3, 256, 256
M = 1000  # Number of Gaussians

# Create example tensors
rendered_img = torch.randn(B, C, H, W)
style_img = torch.randn(B, C, H, W)
content_img = torch.randn(B, C, H, W)
D_origin = torch.randn(B, 1, H, W)
D_render = torch.randn(B, 1, H, W)
delta_scale = torch.randn(B, M)
delta_opacity = torch.randn(B, M)
F_render = torch.randn(B, 512, H//8, W//8)  # Example VGG features
F_style = torch.randn(B, 512, H//8, W//8)
F_content = torch.randn(B, 512, H//8, W//8)

# Calculate combined loss
total_loss, components = gaussian_stylization_loss(
    rendered_img, style_img, content_img, D_origin, D_render,
    delta_scale, delta_opacity, F_render, F_style, F_content
)
"""



############## below is just perceptual control loss mentioend in the paper ################
def color_control_loss(render_features, style_features):
    """
    Args:
        render_features: Rendered image features (B, C, H, W)
        style_features: Style image features (B, C, H, W)
    """
    Y_render = 0.299 * render_features[:,0:1] + 0.587 * render_features[:,1:2] + 0.114 * render_features[:,2:3]
    Y_style = 0.299 * style_features[:,0:1] + 0.587 * style_features[:,1:2] + 0.114 * style_features[:,2:3]
    return nn.MSELoss()(Y_render, Y_style)

def scale_control_loss(vgg_features_render, vgg_features_style):
    """
    Args:
        vgg_features_render: Dict of VGG features from render
        vgg_features_style: Dict of VGG features from style
    """
    loss = 0
    layer_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.6, 'conv4_1': 0.4}
    
    for layer, weight in layer_weights.items():
        F_l = vgg_features_render[layer]
        F_s = vgg_features_style[layer]
        loss += weight * torch.mean((F_l - F_s) ** 2)
        
    return loss

def spatial_control_loss(render_features, style_features, masks_render, masks_style):
    """
    Args:
        render_features: Features (B, C, H, W)  
        style_features: Features (B, C, H, W)
        masks_render: Region masks (B, R, H, W)
        masks_style: Region masks (B, R, H, W)
    """
    loss = 0
    num_regions = masks_render.shape[1]
    
    for r in range(num_regions):
        m_r = masks_render[:,r:r+1]
        m_s = masks_style[:,r:r+1]
        
        masked_render = render_features * m_r
        masked_style = style_features * m_s
        
        loss += torch.mean((masked_render - masked_style) ** 2)
        
    return loss / num_regions