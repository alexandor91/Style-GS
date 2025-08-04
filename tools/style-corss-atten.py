import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests


# Define AdaIN, CrossAttentionBlock, and GaussianStyleTransformer
class AdaIN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.style_scale = nn.Parameter(torch.ones(dim))
        self.style_bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        return self.style_scale * self.norm(x) + self.style_bias


class CrossAttentionBlock(nn.Module):
    def __init__(self, sh_dim=48, dino_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = sh_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(sh_dim, sh_dim)
        self.k_proj = nn.Linear(dino_dim, sh_dim)
        self.v_proj = nn.Linear(dino_dim, sh_dim)
        self.out_proj = nn.Linear(sh_dim, sh_dim)
        
        self.ada_in = AdaIN(sh_dim)
        
    def forward(self, sh_features, style_tokens):
        B, N, C = sh_features.shape  # Batch size, tokens in SH, feature dimension
        H = self.num_heads           # Number of heads
        T = style_tokens.shape[1]    # Number of style tokens

        # Query, Key, Value projections
        q_proj = self.q_proj(sh_features)  # Shape: [B, N, sh_dim]
        q = q_proj.view(B, N, H, self.head_dim).permute(0, 2, 1, 3)  # Shape: [B, H, N, head_dim]

        k_proj = self.k_proj(style_tokens)  # Shape: [B, T, sh_dim]
        print("@@@@@@@@@@@@@@@@@@@@@@@")
        print(k_proj.shape)
        assert k_proj.shape[2] == H * self.head_dim, f"sh_dim mismatch: {k_proj.shape[2]} != {H * self.head_dim}"
        k = k_proj.reshape(B, T, H, self.head_dim).permute(0, 2, 1, 3)  # Shape: [B, H, T, head_dim]

        v_proj = self.v_proj(style_tokens)  # Shape: [B, T, sh_dim]
        v = v_proj.view(B, T, H, self.head_dim).permute(0, 2, 1, 3)  # Shape: [B, H, T, head_dim]

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Shape: [B, H, N, T]
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Shape: [B, N, sh_dim]
        out = self.out_proj(out)                          # Shape: [B, N, sh_dim]
        out = self.ada_in(out)                            # Shape: [B, N, sh_dim]

        return out



class GaussianStyleTransformer(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.sh_dim = 48  # SH dimension (16 coeffs per RGB channel)
        
        # Load DINOv2 model and processor
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino = AutoModel.from_pretrained('facebook/dinov2-base')
        
        # Freeze DINO weights
        for param in self.dino.parameters():
            param.requires_grad = False
            
        self.layers = nn.ModuleList([
            CrossAttentionBlock(sh_dim=self.sh_dim, dino_dim=768)
            for _ in range(num_layers)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(self.sh_dim, self.sh_dim * 2),
            nn.ReLU(),
            nn.Linear(self.sh_dim * 2, self.sh_dim)
        )
        
    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    
    def forward(self, gaussian_sh, style_image):
        # Process style image
        style_inputs = self.preprocess_image(style_image).to(next(self.dino.parameters()).device)

        with torch.no_grad():
            style_features = self.dino(pixel_values=style_inputs).last_hidden_state  # Shape: [B, 257, 768]

        x = gaussian_sh.to(next(self.dino.parameters()).device)  # Ensure Gaussian SH is on the same device
        for layer in self.layers:
            x = x + layer(x, style_features)

        return self.mlp(x)


# Define loss computation
def compute_feature_loss(model, input_sh, output_sh, style_image):
    """
    Compute the content and style loss for training.

    Args:
        model (GaussianStyleTransformer): The model being trained.
        input_sh (torch.Tensor): The input SH features of shape [batch_size, N, 48].
        output_sh (torch.Tensor): The output SH features of shape [batch_size, N, 48].
        style_image (torch.Tensor): The style image tensor of shape [batch_size, 3, H, W].

    Returns:
        tuple: Content loss and style loss.
    """
    # Process style image
    with torch.no_grad():
        # Process the style image
        style_inputs = model.preprocess_image(style_image).to(next(model.dino.parameters()).device)

        # Pass the processed image to the DINO model
        style_features = model.dino(pixel_values=style_inputs).last_hidden_state  # Shape: [batch_size, 256+1, 768]
    # Compute content loss
    content_loss = F.mse_loss(output_sh, input_sh)

    # Project DINO features to SH dimension (48)
    style_mean = style_features.mean(dim=1)  # Shape: [batch_size, 768]
    style_std = style_features.std(dim=1)    # Shape: [batch_size, 768]

    # Define a linear projection layer (shared across batches)
    projection = nn.Linear(768, 48).to(style_mean.device)  # Map from 768 -> 48
    projected_style_mean = projection(style_mean)          # Shape: [batch_size, 48]
    projected_style_std = projection(style_std)            # Shape: [batch_size, 48]

    # Compute mean and std for the output SH
    output_mean = output_sh.mean(dim=1)  # Shape: [batch_size, 48]
    output_std = output_sh.std(dim=1)    # Shape: [batch_size, 48]

    # Compute style loss
    style_loss = F.mse_loss(output_mean, projected_style_mean) + F.mse_loss(output_std, projected_style_std)

    return content_loss, style_loss


# Training loop
def train_iteration(model, optimizer, gaussian_sh, style_image, content_weight=1.0, style_weight=0.1):
    start_time = time.time()
    
    optimizer.zero_grad()
    output_sh = model(gaussian_sh, style_image)
    
    content_loss, style_loss = compute_feature_loss(model, gaussian_sh, output_sh, style_image)
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # Backward pass
    backward_start_time = time.time()  # Start backward timing
    total_loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters
    backward_time = time.time() - backward_start_time  # Backward pass duration

    print("$$$$$$$$$$ backward iteration time $$$$$$$$$$$")
    print(backward_time)    
    iteration_time = time.time() - start_time
    
    return {
        'total_loss': total_loss.item(),
        'content_loss': content_loss.item(),
        'style_loss': style_loss.item(),
        'iteration_time': iteration_time
    }


# Main training script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GaussianStyleTransformer(num_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Dummy inputs for Gaussian SH coefficients and style image
    gaussian_sh = torch.randn(1, 100000, 48).to(device)  # Example batch_size = 1
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    style_image = Image.open(requests.get(url, stream=True).raw)

    # Training
    num_iterations = 1000
    for i in range(num_iterations):
        metrics = train_iteration(model, optimizer, gaussian_sh, style_image)
        print(f"Iteration {i + 1}: Total Loss = {metrics['total_loss']:.4f}, "
            f"Content Loss = {metrics['content_loss']:.4f}, "
            f"Style Loss = {metrics['style_loss']:.4f}, "
            f"Iteration Time = {metrics['iteration_time']:.4f} seconds")