import torchvision.models as models
import torch.nn as nn

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True, model_name='resnet18'):
        super(ResNetEncoder, self).__init__()
        if model_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        # Remove the final pooling and fc layers; output shape will be [B,512,8,8] for a 256x256 input.
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        # Optionally freeze encoder parameters:
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)