import torch.nn as nn

from graph.weights_initializer import weights_init


def encoder_conv(_in, _out):
    return nn.Sequential(
        nn.Conv2d(_in, _out, 3, 1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.modules = nn.ModuleList([
            encoder_conv(6, 32),
            encoder_conv(32, 64),
            encoder_conv(64, 128),
            encoder_conv(128, 256),
            encoder_conv(256, 512),
            encoder_conv(512, 1024),
            encoder_conv(1024, 2048),
        ])

        self.apply(weights_init)

    def forward(self, x):
        """
        :param x: 6 x 512 x 1024
        :return: small module output list
        ex) conv4 => 256 x 32 x 64
            conv7 => 2048 x 4 x 8
        """
        out_list = []

        for conv in self.modules:
            x = conv(x)
            out_list.append(x)
        return out_list
