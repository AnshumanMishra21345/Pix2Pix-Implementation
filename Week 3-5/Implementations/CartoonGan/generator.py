import torch.nn as nn
import torch
from pretrained_resnet_encoder import ResNetEncoder


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        return x + self.block(x)

class CartoonGenerator(nn.Module):
    def __init__(self, num_res_blocks=8, resnet_model='resnet18'):
        super(CartoonGenerator, self).__init__()
        self.encoder = ResNetEncoder(model_name=resnet_model)
        # For resnet18, the encoder outputs features with shape [B,512,8,8]
        channels = 512
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )
        
        # Modified Decoder: 5 upsampling blocks to go from 8x8 -> 256x256.
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(channels, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output normalized between -1 and 1.
        )

    def forward(self, x):
        encoded_features = self.encoder(x)      # Expect shape: [B,512,8,8]
        processed_features = self.res_blocks(encoded_features)
        cartoon_image = self.decoder(processed_features)  # Now output is [B,3,256,256]
        return cartoon_image