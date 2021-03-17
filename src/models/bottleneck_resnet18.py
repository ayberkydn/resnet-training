import torch
import numpy as np
import torchvision



class BottleneckResnet18(torch.nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.bottleneck = torch.Conv2d(3, 1, 1, bias=False)
        self.resnet = torchvision.models.resnet18(pretrained=False)
    def forward(self, x):

        x = self.pre_conv(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)

        x = self.post_conv(x)
        return x

if __name__ == "__main__":
    data = torch.randn(1, 3, 224, 224, device="cuda")
    net = BottleneckResnet18().to("cuda")
    out = net(data)
