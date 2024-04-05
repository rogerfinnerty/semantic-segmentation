"""
CS585 HW4 Semantic Segmentation
Roger Finnerty, Demetrios Kechris, Benjamin Burnham
April 1, 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models
from torchvision.transforms import v2


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass"""
        out_1 = self.features_block1(x) # (H/2, W/2)
        out_2 = self.features_block2(out_1) # (H/4, W/4)
        out_3 = self.features_block3(out_2) # (H/8, W/8)
        out_4 = self.features_block4(out_3) # (H/16, W/16)
        out_5 = self.features_block5(out_4) # (H/32, W/32)

        score_5 = self.score_pool4(out_5)   # (H/32, W/32)

        upscore2 = self.upscore2(score_5)   # (H/16, H/16)
        score_4 = self.score_pool4(out_4) # (H/16, W/16)
        upscore2 = v2.CenterCrop(size=(24, 32))(upscore2)

        upscore4 = self.upscore_pool4(upscore2 + score_4) # (H/8, W/8)
        score_3 = self.score_pool3(out_3) # (H/8, W/8)
        upscore4 = v2.CenterCrop(size=(48, 64))(upscore4)

        upscore_final = self.upscore_final(score_3 + upscore4) # (H, W)
        upscore_final = v2.CenterCrop(size=(384, 512))(upscore_final)

        return upscore_final
