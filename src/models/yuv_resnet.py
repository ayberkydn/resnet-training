import torch, torchvision
import numpy as np
import kornia
import pdb

class YUVResnet(torch.nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        self.luma_resnet = torchvision.models.resnet18(pretrained=False)
        self.chroma_resnet = torchvision.models.resnet18(pretrained=False)

        self.luma_resnet.conv1   = torch.nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
        self.chroma_resnet.conv1 = torch.nn.Conv2d(2, 64, (7,7), (2,2), (3,3), bias=False)

        self.chroma_resnet.fc = torch.nn.Identity()
        self.luma_resnet.fc = torch.nn.Identity()

        self.fc = torch.nn.Linear(1024, out_classes)


    def forward(self, x):
        yuv_x = kornia.rgb_to_yuv(x)

        x_chroma = yuv_x[:, 1:, ...]
        x_luma = yuv_x[:, 0:1, ...]

        chroma_features = self.chroma_resnet(x_chroma)
        luma_features = self.luma_resnet(x_luma)

        features = torch.cat([chroma_features, luma_features], dim=1)

        return self.fc(features)




if __name__ == "__main__":
    data = torch.randn(8, 3, 224, 224, device="cuda")
    net = YUVResnet(out_classes=10).to('cuda')
    print(net(data).shape)
