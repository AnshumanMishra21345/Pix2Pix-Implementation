import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name='conv4_4'):
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        # Here, we extract features up to the 21st layer (approximately conv4_4)
        self.feature_extractor = nn.Sequential(*list(vgg19.children())[:21])
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)