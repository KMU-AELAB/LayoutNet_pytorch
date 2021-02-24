import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Encoder, Edge, Corner, Regressor

from graph.weights_initializer import weights_init


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = Encoder()
        self.edge = Edge()
        self.corner = Corner()
        self.regressor = Regressor()

        self.apply(weights_init)

    def forward(self, x):
        encoder_out_list = self.encoder(x)
        edge_out_list, edge_out = self.edge(encoder_out_list)
        corner_out = self.corner(edge_out_list)

        reg_out = self.regressor(torch.cat((corner_out, edge_out), dim=1))

        return edge_out, corner_out, reg_out


if __name__ == '__main__':
    from torchsummary import summary

    model = Model()
    summary(model, (6, 512, 1024), batch_size=10)