import torch.nn as nn

class LocalNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Image size is 33 x 33 x 3, outputs feature map size 1 x 1 x 128
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.005),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(128, 256, 5, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.005),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(256, 256, 2, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.005),
            nn.Conv2d(256, 128, 4, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.005)
        )

    def forward(self, x):
        x = self.model(x)
        return x