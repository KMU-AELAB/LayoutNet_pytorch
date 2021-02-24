import torch
import torch.nn as nn

from graph.model.encoder import Encoder
from graph.model.decoder import Edge, Corner

from graph.weights_initializer import weights_init


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = Encoder()
        self.edge = Edge()
        self.corner = Corner()

        self.apply(weights_init)

    def forward(self, x):
        encoder_out_list = self.encoder(x)
        edge_out_list, edge_out = self.edge(encoder_out_list)
        corner_out = self.corner(edge_out_list)

        return edge_out, corner_out
