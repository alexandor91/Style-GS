import torch
from torchvision import transforms
import glob
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# DINOv2
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.cuda()
model.eval()
out_indices = [2, 5, 8, 11]
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True
        ),
    ]
)
"""
遍历images文件夹下所有图像，保存对应的cls feature到featrues文件夹
只要保存每个layer的cls token应该就行
"""
image_paths = glob.glob("./images/**/*.jpg", recursive=True)
N = len(image_paths)
for i in tqdm(range(N)):
    # opencv deault BGR
    image_path = image_paths[i]
    feature_path = image_path.replace("images", "features")
    feature_path = feature_path.replace(".jpg", ".npz")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    with torch.no_grad():
        image = transform(image).unsqueeze(0).float().cuda()
        _, _, H, W = image.shape
        image = torch.nn.functional.interpolate(
            image, size=(14 * (H // 14), 14 * (W // 14)), mode="bicubic"
        )
        result = model.get_intermediate_layers(
            image,
            reshape=True,
            n=out_indices,
            return_class_token=True,
        )
        np_dict = {}
        for i in range(len(result)):
            layer_indice = out_indices[i]
            np_dict[f"layer_{layer_indice}_feature_map"] = (
                result[i][0].squeeze(0).cpu().numpy()
            )
            np_dict[f"layer_{layer_indice}_cls_token"] = (
                result[i][1].squeeze(0).cpu().numpy()
            )

    if not os.path.exists(os.path.split(feature_path)[0]):
        os.makedirs(os.path.split(feature_path)[0])

    np.savez_compressed(feature_path, **np_dict)
    # test_np_dict = np.load(feature_path)
    # import pdb

    # pdb.set_trace()
    # import pdb

    # pdb.set_trace()
