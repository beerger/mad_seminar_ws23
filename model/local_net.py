import torch.nn as nn

class LocalNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        #self.device = torch.device(config['device']) # Set device as defined in yaml-file

        # TODO: Should bias be true or false?
        # Image size is 33 x 33 x 3, outputs feature map size 1 x 1 x 128
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=self.config['leaky_relu_slope']),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(128, 256, 5, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=self.config['leaky_relu_slope']),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(256, 256, 2, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=self.config['leaky_relu_slope']),
            nn.Conv2d(256, 128, 4, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=self.config['leaky_relu_slope'])
        )

        #self.model.to(self.device) # Move the model to the specified device

    def forward(self, x):
        x = self.model(x)
        return x