import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self,config):
        self.root_dir = config.root_path
        self.data_list = os.listdir(os.path.join(self.root_dir, config.data_path))
        self.config = config

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = os.path.join(self.root_dir, self.config.data_path, 'train', self.data_list[idx])
        _data = np.load(data_path)

        data = [_data['img'], _data['line'], _data['corner'], _data['edge'], _data['box']]

        # flipping
        if random.random() < 0.6:
            for i in range(len(data)):
                data[i] = np.flip(data[i], axis=2)

        # rolling
        if random.random() < 0.6:
            dx = np.random.randint(data[0].shape[2])
            for i in range(len(data)):
                data[i] = np.roll(data[i], dx, axis=2)

        # gamma augmentation
        if random.random() < 0.6:
            p = np.random.uniform(0.5, 2)
            data[0] = data[0] ** p

        return {'img': data[0], 'line': data[1], 'corner': data[2], 'edge': data[3], 'box': data[4]}
