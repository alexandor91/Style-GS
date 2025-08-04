import os
import json
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision import models
from collections import OrderedDict
from typing import Sequence
# Import utility functions
from utils.image_utils import psnr
# from utils.loss_utils import ssim
from lpipsPyTorch import lpips

def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False



class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)

def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict

class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)
    

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# def ssim(img1, img2, window_size=11, size_average=True):
#     channel = img1.size(-3)

#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)

#     return _ssim(img1, img2, window, window_size, channel, size_average)

# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
    
def read_images(output_dir, gt_dir):
    """Reads and loads images from specified directories, ensuring sorted order."""
    output_files = sorted(os.listdir(output_dir))
    gt_files = sorted(os.listdir(gt_dir))
    
    renders, gts, image_names = [], [], []
    for out_file, gt_file in zip(output_files, gt_files):
        if out_file == gt_file:  # Ensure corresponding images
            render = Image.open(output_dir / out_file).convert("RGB")
            gt = Image.open(gt_dir / gt_file).convert("RGB")
            
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(out_file)
    return renders, gts, image_names

def compute_rmse(img1, img2):
    """Computes RMSE between two images."""
    return torch.sqrt(torch.mean((img1 - img2) ** 2)).item()

def evaluate(output_dir, gt_dir):
    """Evaluates LPIPS, RMSE, PSNR, and SSIM between rendered and ground truth images."""
    renders, gts, image_names = read_images(output_dir, gt_dir)
    
    results = {"PSNR": {}, "LPIPS": {}, "RMSE": {}}
    
    ssims, psnrs, lpipss, rmses = [], [], [], []
    
    for idx in tqdm(range(len(renders)), desc="Evaluating Metrics"):
        # ssim_val = ssim(renders[idx], gts[idx])
        psnr_val = psnr(renders[idx], gts[idx])
        lpips_val = lpips(renders[idx], gts[idx], net_type='vgg')
        rmse_val = compute_rmse(renders[idx], gts[idx])
        
        # ssims.append(ssim_val)
        psnrs.append(psnr_val)
        lpipss.append(lpips_val)
        rmses.append(rmse_val)
        
        # results["SSIM"][image_names[idx]] = ssim_val
        results["PSNR"][image_names[idx]] = psnr_val
        results["LPIPS"][image_names[idx]] = lpips_val.item()
        results["RMSE"][image_names[idx]] = rmse_val
    
    # Compute overall metrics
    results["Overall"] = {
        # "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
        "RMSE": torch.tensor(rmses).mean().item(),
    }
    
    return results

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate GS Rendered Stylized Images Against Ground Truth")
    parser.add_argument('--output_dir', '-o', required=True, type=str, help="Directory containing rendered images")
    parser.add_argument('--gt_dir', '-g', required=True, type=str, help="Directory containing ground truth images")
    parser.add_argument('--output_json', '-j', required=True, type=str, help="Path to save evaluation results as JSON")
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    results = evaluate(Path(args.output_dir), Path(args.gt_dir))
    
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Evaluation complete. Results saved to", args.output_json)
