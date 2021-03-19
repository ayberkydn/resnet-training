import torch
import numpy as np
import torchvision
import einops


class BottleneckLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck = torch.nn.Conv2d(in_channels, 1, 1, bias=False)

    def forward(self, x):
        x = self.bottleneck(x)
        x = torch.squeeze(x, 1)
        x = einops.repeat(x, 'b h w -> b c h w', c=self.in_channels)
        return x

class BottleneckResnet50(torch.nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()

        self.bottleneck = BottleneckLayer(in_channels)
        self.resnet = torchvision.models.resnet50(pretrained=False)
        self.resnet.fc = torch.nn.Linear(
            self.resnet.fc.in_features,
            out_dim,
        )

    def forward(self, x):

        x = self.bottleneck(x)
        x = self.resnet(x)
        return x

if __name__ == "__main__":
    data = torch.randn(1, 3, 224, 224, device="cuda")
    net = BottleneckResnet50(3, 10).to("cuda")
    out = net(data)
