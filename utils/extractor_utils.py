import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms
import os
import sys

sys.path.append(os.getcwd())

from dinov2.models.vision_transformer import vit_small


"""Implement feature extractor"""


class VGG19Extractor(nn.Module):
    def __init__(
        self,
        out_indices=[2, 7, 16, 25, 34],
        device="cuda:0",
    ) -> None:
        super().__init__()
        # scale factor 32

        self.model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
        self.out_indices = out_indices
        self.normalization = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True
        )

        self.model.cuda()
        self.model.eval()
        
        # 全部先resize到224才是有效的

    def forward(self, x, size):
        h, w = size
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        x = self.normalization(x)
        outs = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return outs

    def get_input_size(self, img):
        h, w = img.shape[-2], img.shape[-1]
        h = (h // 32) * 32
        w = (w // 32) * 32
        return h, w
        # return 224,224


# from dinov2.models.vision_transformer import vit_small


class DINOV2SExtractor(nn.Module):
    def __init__(self, out_indices=[2, 5, 8, 11]):
        super().__init__()
        self.out_indices = out_indices
        self.normalization = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True
        )
        # self.model = torch.hub.load(
        #     "facebookresearch/dinov2", "dinov2_vits14", skip_validation=True
        # )
        self.model = vit_small(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        self.model.load_state_dict(
            torch.load(
                os.path.join(os.getcwd(), "dinov2", "dinov2_vits14_pretrain.pth")
            )
        )
        self.model.cuda()
        self.model.eval()

    def forward(self, x, size):
        h, w = size
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        x = self.normalization(x)
        result = self.model.get_intermediate_layers(
            x,
            reshape=True,
            n=self.out_indices,
            return_class_token=True,
        )
        return result
        # result_dict = []
        # for i in range(len(result)):
        #     layer_indice = self.out_indices[i]
        #     result_dict[f"layer_{layer_indice}_feature_map"] = (
        #         result[i][0].squeeze(0).cpu().numpy()
        #     )
        #     result_dict[f"layer_{layer_indice}_cls_token"] = (
        #         result[i][1].squeeze(0).cpu().numpy()
        #     )
        # return result_dict

    def get_input_size(self, img):
        h, w = img.shape[-2], img.shape[-1]
        h = (h // 14) * 14
        w = (w // 14) * 14
        return h, w


if __name__ == "__main__":
    test = VGG19Extractor()
    img = cv2.imread("images/1.jpg")
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
    img = img.unsqueeze(0).float().cuda()
    res = test(img, (224, 224))
    import pdb

    pdb.set_trace()

# style IN Loss实现？ 应该就是各层的feature的一个loss
