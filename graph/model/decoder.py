import torch
import torch.nn as nn

from graph.weights_initializer import weights_init


def decoder_conv(_in, _out):
    return nn.Sequential(
        nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
    )


class Edge(nn.Module):
    def __init__(self):
        super(Edge, self).__init__()

        self.module_lst = nn.ModuleList([
            decoder_conv(2048, 1024),
            decoder_conv(1024 * 2, 512),
            decoder_conv(512 * 2, 256),
            decoder_conv(256 * 2, 128),
            decoder_conv(128 * 2, 64),
            decoder_conv(64 * 2, 32),
        ])

        self.conv = nn.Conv2d(32*2, 3, 3, 1)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        _input = x[-1]
        out_list = [_input]

        for idx, conv in enumerate(self.module_lst):
            _input = nn.functional.interpolate(_input, scale_factor=2, mode='nearest')
            _input = conv(_input)
            _input = torch.cat((x[-(idx + 2)], _input), dim=1)
            out_list.append(_input)

        _input = nn.functional.interpolate(_input, scale_factor=2, mode='nearest')
        _input = self.conv(_input)
        out = self.sigmoid(_input)

        return out_list, out


class Corner(nn.Module):
    def __init__(self):
        super(Corner, self).__init__()

        self.module_lst = nn.ModuleList([
            decoder_conv(2048, 1024),
            decoder_conv(1024 * 3, 512),
            decoder_conv(512 * 3, 256),
            decoder_conv(256 * 3, 128),
            decoder_conv(128 * 3, 64),
            decoder_conv(64 * 3, 32),
        ])

        self.conv = nn.Conv2d(32 * 3, 1, 3, 1)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        _input = x[0]

        for idx, conv in enumerate(self.module_lst):
            _input = nn.functional.interpolate(_input, scale_factor=2, mode='nearest')
            _input = conv(_input)
            _input = torch.cat((x[(idx + 1)], _input), dim=1)

        _input = nn.functional.interpolate(_input, scale_factor=2, mode='nearest')
        _input = self.conv(_input)
        out = self.sigmoid(_input)

        return out