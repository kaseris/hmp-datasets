import pickle
import re

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
        self.unique_names = set()
        names_filtered = list(map(self.remove_subscripts_and_extension, self.names))
        for name in names_filtered:
            self.unique_names.add(name)
        self.name2class = {k: v for v, k in enumerate(self.unique_names)}
        self.motion_data = list(self.data.values())

    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.name2class[self.remove_subscripts_and_extension(name)]
        label = torch.tensor(label)
        motion_data = torch.from_numpy(self.motion_data[idx])
        if self.return_name:
            return motion_data, label, name
        return motion_data, label

    def __len__(self):
        return len(self.motion_data)
    
    def remove_subscripts_and_extension(self, x: str)->str:
        return re.sub(r"_\d+.*", "", x)