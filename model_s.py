import torch
import torch.nn as nn


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
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_0 = DoubleConv(3, 128)
        self.up_1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv_1 = DoubleConv(131, 131)
        self.up_2 = nn.ConvTranspose2d(131, 131, 2, stride=2)
        self.conv_2 = DoubleConv(134, 134)
        self.output = nn.Conv2d(134, 3, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):  # [1, 3, 256, 256]
        x_2 = self.upsample(x)
        x_4 = self.upsample(x_2)

        x = self.conv_0(x)  # [1, 64, 256, 256]

        x = self.up_1(x)  # [1, 64, 512, 512]
        x = torch.concat((x_2, x), dim=1)
        x = self.conv_1(x)  # [1, 64, 512, 512]

        x = self.up_2(x)  # [1, 64, 1024, 1024]
        x = torch.concat((x_4, x), dim=1)
        x = self.conv_2(x)  # [1, 64, 1024, 1024]

        x = self.output(x)  # [1, 3, 1024, 1024]
        return x


class Descriminator(nn.Module):
    def __init__(self):
        super(Descriminator, self).__init__()

        self.conv1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(64, 64)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(64, 64)
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
    inputs = torch.rand(1, 3, 512, 240)
    net = Generator()
    outputs = net.forward(inputs)
    print(outputs.size())
