import torch
import numpy as np
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, out, tg1, tg2, gt3):
        return self.bce(out[0], tg1) + self.bce(out[1], tg2) + self.mse(out[2], gt3)
