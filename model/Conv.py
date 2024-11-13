from torch import nn


class MyConv(nn.Module):
    def __init__(self, img_size, in_channels):
        super(MyConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1) # shape
        self.conv2 = nn.Conv2d(64, 32, kernel_size=10, stride=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=10, stride=1)
        width = img_size - 20
        height = img_size - 20
        self.l1 = nn.Linear(width * height, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.l1(x)
        return x