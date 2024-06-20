import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = use_res_connect
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            # nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        y = self.conv(x)
        return x+y


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.block_1 = InvertedResidual(3, 128, 1, False)
        self.up_1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.block_2 = InvertedResidual(131, 131, 1, True)
        self.block_3 = InvertedResidual(131, 131, 1, True)
        self.up_2 = nn.ConvTranspose2d(131, 131, 2, stride=2)
        self.block_4 = InvertedResidual(134, 134, 1, True)
        self.block_5 = InvertedResidual(134, 134, 1, True)
        self.output = InvertedResidual(134, 3, 1, False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x2 = self.upsample(x)
        x4 = self.upsample(x2)

        x = self.block_1(x)

        x = self.up_1(x)
        x = torch.concat((x2, x), dim=1)
        x = self.block_2(x)
        x = self.block_3(x)

        x = self.up_2(x)
        x = torch.concat((x4, x), dim=1)
        x = self.block_4(x)
        x = self.block_5(x)

        x = self.output(x)
        return x


class Descriminator(nn.Module):
    def __init__(self):
        super(Descriminator, self).__init__()
        self.conv1 = InvertedResidual(3, 16, 1, False)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = InvertedResidual(16, 32, 1, False)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = InvertedResidual(32, 64, 1, False)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = InvertedResidual(64, 64, 1, True)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = InvertedResidual(64, 64, 1, True)
        self.fc6 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)  # [1, 32, 160, 120]
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)  # [1, 64, 20, 15]
        x = self.adaptive_pool(conv5)  # [1, 64, 1, 1]
        x = x.view(-1,)
        x = self.fc6(x)
        logits = self.sigmoid(x)
        return logits


if __name__ == "__main__":
    from thop import profile
    inputs = torch.rand(1, 3, 256, 256)
    model = Generator()
    outputs = model.forward(inputs)
    print(outputs.size())
    # flops, params = profile(model, inputs=(input, ))
    # print(f'FLOPs: {flops}')
    # print(f'Params: {params}')
