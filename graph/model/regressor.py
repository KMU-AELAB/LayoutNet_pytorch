import torch
import torch.nn as nn

from graph.weights_initializer import weights_init


def encoder_conv(_in, _out):
    return nn.Sequential(
        nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.module_lst = nn.ModuleList([
            encoder_conv(4, 8),
            encoder_conv(8, 16),
            encoder_conv(16, 32),
            encoder_conv(32, 64),
            encoder_conv(64, 128),
            encoder_conv(128, 256),
            encoder_conv(256, 512),
        ])

        self.linear1 = nn.Linear(512 * 4 * 8, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, 6)

        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        """
        :param x: 4 x 512 x 1024
        :return: 1 x 6
        """

        for conv in self.module_lst:
            x = conv(x)
        x = x.view(-1, 512 * 4 * 8)

        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        out = self.relu(self.linear4(x))

        return out
