import pickle

import torch

from torch.utils.data import Dataset


class RokokoDataset(Dataset):
    def __init__(self, data_path: str, return_name: bool = False):
        self.data_path = data_path
        self.return_name = return_name
        try:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        except Exception as e:
            print(e)
        self.names = list(self.data.keys())
        self.motion_data = list(self.data.values())

    def __getitem__(self, idx):
        return self.motion_data[idx]

    def __len__(self):
        return len(self.motion_data)
    