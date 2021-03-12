
def bn_relu_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    )

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel_size, out_channel_sizes, strides, kernels, paddings):
        super().__init__()

        output_size = out_channel_sizes[-1]
        if in_channel_size != output_size:
            self.residual_connection = torch.nn.Conv2d(in_channel_size, output_size, 1, stride=np.prod(strides))
        else:
            self.residual_connection = torch.nn.Identity()

        channel_sizes = [in_channel_size] + out_channel_sizes

        layers = [bn_relu_conv(channel_sizes[n], channel_sizes[n+1], kernels[n], strides[n], paddings[n]) for n in range(3)]
        self.layers = torch.nn.Sequential(*layers)



    def forward(self, x):

        #print('x', x.shape)
        #print('layers', self.layers(x).shape)
        #print('residual', self.residual_connection(x).shape)
        return self.layers(x) + self.residual_connection(x)

class MultiResidualBlock(torch.nn.Module):
    def __init__(self, count, in_channel_size, out_channel_sizes, strides, kernels, paddings):
        super().__init__()
        blocks = []
        for n in range(count):
            blocks.append(ResidualBlock(in_channel_size, out_channel_sizes, strides, kernels, paddings))
            if n == 0: #after the first block, all strides are 1 and output shape == input shape
                strides = [1] * len(strides)
                in_channel_size = out_channel_sizes[-1]
        self.layers = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)

    def forward(self, x):

        #print('x', x.shape)
        #print('layers', self.layers(x).shape)
        #print('residual', self.residual_connection(x).shape)
        return self.layers(x)

class Resnet50(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()

        self.pre_conv = torch.nn.Sequential(
            bn_relu_conv(3, 64, 7, 2, 3),
            torch.nn.MaxPool2d(3, 2, 1)
        )
        block1_channels = [64, 64, 256]
        block1_strides  = [1, 1, 1]
        block1_kernels  = [1, 3, 1]       
        block1_paddings = [0, 1, 0] 
        block1_layer_count = 3 

        self.conv1 = MultiResidualBlock(block1_layer_count, 
                                        64, 
                                        block1_channels, 
                                        block1_strides, 
                                        block1_kernels, 
                                        block1_paddings)

        block2_channels = [128, 128, 512]
        block2_strides  = [2, 1, 1]
        block2_kernels  = [1, 3, 1]       
        block2_paddings = [0, 1, 0] 
        block2_layer_count = 4
        
        self.conv2 = MultiResidualBlock(block2_layer_count, 
                                        block1_channels[-1], 
                                        block2_channels, 
                                        block2_strides, 
                                        block2_kernels, 
                                        block2_paddings)

        block3_channels = [256, 256, 1024]
        block3_strides  = [2, 1, 1]
        block3_kernels  = [1, 3, 1]       
        block3_paddings = [0, 1, 0] 
        block3_layer_count = 6
        
        self.conv3 = MultiResidualBlock(block3_layer_count, 
                                        block2_channels[-1], 
                                        block3_channels, 
                                        block3_strides, 
                                        block3_kernels, 
                                        block3_paddings)

        block4_channels = [512, 512, 2048]
        block4_strides  = [2, 1, 1]
        block4_kernels  = [1, 3, 1]       
        block4_paddings = [0, 1, 0] 
        block4_layer_count = 3
        
        self.conv4 = MultiResidualBlock(block4_layer_count,
                                        block3_channels[-1], 
                                        block4_channels, 
                                        block4_strides, 
                                        block4_kernels, 
                                        block4_paddings)
        
        
    
        
        
        self.post_conv = torch.nn.Sequential(
            torch.nn.AvgPool2d(7),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 1000)
        )

    def forward(self, x):
        
        x = self.pre_conv(x)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.conv4(x)
        
        x = self.post_conv(x)
        return x
        

data = torch.randn(1, 3, 224, 224, device='cuda')
net =  Resnet50().to('cuda')
out = net(data)
