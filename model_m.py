import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.conv1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(64, 64)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(64, 64)
        # Decoder
        self.up6 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv6 = DoubleConv(128, 64)
        self.up7 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv7 = DoubleConv(128, 64)
        self.up8 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv8 = DoubleConv(96, 32)
        self.up9 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv9 = DoubleConv(32, 32)

        self.up10 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.conv10 = DoubleConv(64, 64)

        self.up11 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv11 = DoubleConv(64, 16)

        self.output = nn.Conv2d(16, 3, 1)

    def forward(self, x):  # 1, 3, 256, 256]
        conv1 = self.conv1(x)  # [1, 32, 256, 256]
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)  # [1, 64, 128, 128]
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)  # [1, 64, 64, 64]
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)  # [1, 64, 32, 32]
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)  # [1, 64, 16, 16]

        up6 = self.up6(conv5)  # [1, 64, 32, 32]
        meger6 = torch.cat([up6, conv4], dim=1)  # [1, 128, 32, 32]
        conv6 = self.conv6(meger6)  # [1, 64, 32, 32]

        up7 = self.up7(conv6)  # [1, 64, 64, 64]
        meger7 = torch.cat([up7, conv3], dim=1)  # [1, 128, 64, 64]
        conv7 = self.conv7(meger7)  # [1, 64, 64, 64]

        up8 = self.up8(conv7)  # [1, 64, 128, 128]
        meger8 = torch.cat([up8, conv2], dim=1)  # [1, 96, 128, 128]
        conv8 = self.conv8(meger8)  # [1, 32, 128, 128]

        up9 = self.up9(conv8)  # [1, 16, 256, 256]
        meger9 = torch.cat([up9, conv1], dim=1)  # [1, 32, 256, 256]
        conv9 = self.conv9(meger9)  # [1, 32, 256, 256]

        up10 = self.up10(conv9)  # [1, 16, 512, 512]
        conv10 = self.conv10(up10)  # [1, 16, 512, 512]

        up11 = self.up11(conv10)  # [1, 16, 512, 512] - [1, 64, 512, 512]
        conv11 = self.conv11(up11)  # [1, 16, 1024, 1024]

        out = self.output(conv11)  # [1, 3, 1024, 1024]
        return out


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
