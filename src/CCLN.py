import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms

class CCLN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_concat1 = nn.Conv2d(2048 + 1024, 512, kernel_size=1)
        self.bn_concat1 = nn.BatchNorm2d(512)
        self.conv_up1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_up1 = nn.BatchNorm2d(512)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_concat2 = nn.Conv2d(512 + 512, 256, kernel_size=1)
        self.bn_concat2 = nn.BatchNorm2d(256)
        self.conv_up2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_up2 = nn.BatchNorm2d(256)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_concat3 = nn.Conv2d(256 + 256, 128, kernel_size=1)
        self.bn_concat3 = nn.BatchNorm2d(128)
        self.conv_up3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_up3 = nn.BatchNorm2d(128)

        self.conv_final = nn.Conv2d(128, 4, kernel_size=1)
        nn.init.constant_(self.conv_final.bias, 0.0)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); r0 = self.maxpool(x)
        r1 = self.layer1(r0); r2 = self.layer2(r1); r3 = self.layer3(r2); r4 = self.layer4(r3)

        x = self.upsample1(r4); x = torch.cat([x, r3], dim=1)
        x = self.leaky_relu(self.bn_concat1(self.conv_concat1(x)))
        x = self.leaky_relu(self.bn_up1(self.conv_up1(x)))

        x = self.upsample2(x); x = torch.cat([x, r2], dim=1)
        x = self.leaky_relu(self.bn_concat2(self.conv_concat2(x)))
        x = self.leaky_relu(self.bn_up2(self.conv_up2(x)))

        x = self.upsample3(x); x = torch.cat([x, r1], dim=1)
        x = self.leaky_relu(self.bn_concat3(self.conv_concat3(x)))
        x = self.leaky_relu(self.bn_up3(self.conv_up3(x)))

        x = self.conv_final(x)
        x = torch.sigmoid(x)
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
    
CCLN_INPUT = (256, 256)

CCLN_TRANSFORM = transforms.Compose([
    transforms.Resize(CCLN_INPUT),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'model', 'state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def load_ccln(path, device):
    model = CCLN().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(_extract_state_dict(ckpt))
    model.eval()
    return model