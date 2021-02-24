import torch
import numpy as np
import torch.nn as nn


class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, out, tg1, tg2, gt3):
        return self.bce(out[0], tg1) + self.bce(out[1], tg2) + (self.mse(out[2], gt3) * 0.01)


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, out, tg):
        return self.bce(out, tg)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, out, tg):
        return self.mse(out, tg)