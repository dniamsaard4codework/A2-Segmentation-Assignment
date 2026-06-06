"""U-Net with a pretrained ResNet-18 encoder — skip vs no-skip ablation.

Both variants share the *identical* ImageNet ResNet-18 encoder and the same
decoder depth/width; the only difference is whether each decoder stage
concatenates the matching encoder feature map (``use_skip``). That isolates the
contribution of skip connections (Exercise 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """Conv → BN → ReLU, twice — the core U-Net block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetResNet18(nn.Module):
    """U-Net / ResNet-18. ``use_skip=False`` removes every encoder→decoder skip.

    Encoder feature maps on a 128×128 input:
        stem_conv → H/2,  64ch   (s0)
        stem_pool → H/4
        layer1    → H/4,  64ch   (s1)
        layer2    → H/8,  128ch  (s2)
        layer3    → H/16, 256ch  (s3)
        layer4    → H/32, 512ch  (s4) → bottleneck
    Decoder upsamples H/32→H over 5 stages.
    """

    def __init__(self, n_classes=3, pretrained=True, use_skip=True):
        super().__init__()
        self.use_skip = use_skip

        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # H/2, 64
        self.stem_pool = resnet.maxpool                                        # H/4
        self.enc1 = resnet.layer1   # H/4,  64
        self.enc2 = resnet.layer2   # H/8,  128
        self.enc3 = resnet.layer3   # H/16, 256
        self.enc4 = resnet.layer4   # H/32, 512

        self.bottleneck = DoubleConv(512, 1024)

        # skip-channel count added to each decoder stage (0 when skips disabled)
        s4, s3, s2, s1, s0 = (512, 256, 128, 64, 64) if use_skip else (0, 0, 0, 0, 0)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(512 + s4, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256 + s3, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128 + s2, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64 + s1, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32 + s0, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def _join(self, x, skip):
        """Concatenate skip (if enabled), upsampling skip to match x if needed."""
        if not self.use_skip:
            return x
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:])
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        s0 = self.stem_conv(x)
        sp = self.stem_pool(s0)
        s1 = self.enc1(sp)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)
        x = self.dec4(self._join(self.up4(x), s4))
        x = self.dec3(self._join(self.up3(x), s3))
        x = self.dec2(self._join(self.up2(x), s2))
        x = self.dec1(self._join(self.up1(x), s1))
        x = self.dec0(self._join(self.up0(x), s0))
        return self.output(x)


def build_segmentation_model(name, n_classes=3, pretrained=True):
    """``unet_resnet18`` → skips on; ``unet_resnet18_no_skip`` → skips off."""
    if name == "unet_resnet18":
        return UNetResNet18(n_classes, pretrained, use_skip=True)
    if name == "unet_resnet18_no_skip":
        return UNetResNet18(n_classes, pretrained, use_skip=False)
    raise ValueError(f"Unknown segmentation model: {name!r}")
