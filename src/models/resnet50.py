import torch, torchvision
import numpy as np


def bn_relu_conv(in_channels, out_channels, kernel_size, stride, padding):
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
    )


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel_size, out_channel_sizes, strides, kernels, paddings):
        super().__init__()

        output_size = out_channel_sizes[-1]
        if in_channel_size != output_size:
            self.residual_connection = torch.nn.Conv2d(
                in_channel_size, output_size, 1, stride=np.prod(strides)
            )
        else:
            self.residual_connection = torch.nn.Identity()

        channel_sizes = [in_channel_size] + out_channel_sizes

        layers = [
            bn_relu_conv(
                channel_sizes[n],
                channel_sizes[n + 1],
                kernels[n],
                strides[n],
                paddings[n],
            )
            for n in range(3)
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + self.residual_connection(x)


class MultiResidualBlock(torch.nn.Module):
    def __init__(
        self, count, in_channel_size, out_channel_sizes, strides, kernels, paddings
    ):
        super().__init__()
        blocks = []
        for n in range(count):
            blocks.append(
                ResidualBlock(
                    in_channel_size, out_channel_sizes, strides, kernels, paddings
                )
            )
            if (
                n == 0
            ):  # after the first block, all strides are 1 and output shape == input shape
                strides = [1] * len(strides)
                in_channel_size = out_channel_sizes[-1]
        self.layers = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


class Resnet50(torch.nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()

        # self.pre_conv = torch.nn.Sequential(
        #     bn_relu_conv(in_channels, 64, 7, 2, 3), torch.nn.MaxPool2d(3, 2, 1)
        # )
        # block1_channels = [64, 64, 256]
        # block1_strides = [1, 1, 1]
        # block1_kernels = [1, 3, 1]
        # block1_paddings = [0, 1, 0]
        # block1_layer_count = 3

        # self.conv1 = MultiResidualBlock(
        #     block1_layer_count,
        #     64,
        #     block1_channels,
        #     block1_strides,
        #     block1_kernels,
        #     block1_paddings,
        # )

        # block2_channels = [128, 128, 512]
        # block2_strides = [2, 1, 1]
        # block2_kernels = [1, 3, 1]
        # block2_paddings = [0, 1, 0]
        # block2_layer_count = 4

        # self.conv2 = MultiResidualBlock(
        #     block2_layer_count,
        #     block1_channels[-1],
        #     block2_channels,
        #     block2_strides,
        #     block2_kernels,
        #     block2_paddings,
        # )

        # block3_channels = [256, 256, 1024]
        # block3_strides = [2, 1, 1]
        # block3_kernels = [1, 3, 1]
        # block3_paddings = [0, 1, 0]
        # block3_layer_count = 6

        # self.conv3 = MultiResidualBlock(
        #     block3_layer_count,
        #     block2_channels[-1],
        #     block3_channels,
        #     block3_strides,
        #     block3_kernels,
        #     block3_paddings,
        # )

        # block4_channels = [512, 512, 2048]
        # block4_strides = [2, 1, 1]
        # block4_kernels = [1, 3, 1]
        # block4_paddings = [0, 1, 0]
        # block4_layer_count = 3

        # self.conv4 = MultiResidualBlock(
        #     block4_layer_count,
        #     block3_channels[-1],
        #     block4_channels,
        #     block4_strides,
        #     block4_kernels,
        #     block4_paddings,
        # )

        # self.post_conv = torch.nn.Sequential(
        #     torch.nn.AvgPool2d(7), torch.nn.Flatten(), torch.nn.Linear(2048, out_classes)
        # )

        # self.layers = torch.nn.Sequential(
        #     self.pre_conv,
        #     self.conv1,
        #     self.conv2,
        #     self.conv3,
        #     self.conv4,
        #     self.post_conv,
        # )

        self.layers = torchvision.models.resnet50(pretrained=False)
        self.layers.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layers.fc = torch.nn.Linear(2048, out_classes)

    def forward(self, x):

        x = self.layers(x)
        return x



if __name__ == "__main__":
    data = torch.randn(1, 2, 224, 224, device="cuda")
    net = Resnet50(in_channels=2,
                   out_classes=10).to("cuda")
    out = net(data)
    print(out.shape)
