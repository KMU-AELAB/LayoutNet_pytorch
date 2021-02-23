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
        data = np.load(data_path)

        # flipping
        if random.random() < 0.6:
            for key in data.keys():
                data[key] = np.flip(data[key], axis=2)

        # rolling
        if random.random() < 0.6:
            dx = np.random.randint(data['img'].shape[2])
            for key in data.keys():
                data[key] = np.roll(data[key], dx, axis=2)

        # gamma augmentation
        if random.random() < 0.6:
            p = np.random.uniform(0.5, 2)
            data['img'] = data['img'] ** p

        return {'img': data['img'], 'line': data['line'], 'box': data['box'],
                'corner': data['corner'], 'edge': data['edge']}
