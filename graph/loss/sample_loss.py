import torch
import numpy as np
import torch.nn as nn


class SampleLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.NLLLoss()

    def forward(self, logits, target):
        return self.loss(logits, target)
