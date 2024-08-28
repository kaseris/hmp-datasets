import pickle
import re

import numpy as np
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from hmpdata.rokoko import ROKOKO_VALID_JOINTS

class MinMaxScaler:
    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        self.min = None
        self.max = None

    def fit(self, sample):
        temp_shape = sample.shape
        sample_reshaped = sample.contiguous().view(-1, 3)
        self.min, _ = sample_reshaped.min(dim=0)
        self.max, _ = sample_reshaped.max(dim=0)

    def transform(self, sample):
        if self.min is None or self.max is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        
        temp_shape = sample.shape
        sample_reshaped = sample.contiguous().view(-1, 3)
        a, b = self.feature_range
        sample_rescaled = a + (sample_reshaped - self.min) * (b - a) / (self.max - self.min)
        return sample_rescaled.view(temp_shape)


class RokokoDataset(Dataset):
    def __init__(self, data_path: str, return_name: bool = False, use_valid_joints=False, transform=False):
        self.data_path = data_path
        self.return_name = return_name
        self.use_valid_joints = use_valid_joints
        self.transform = transform

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
        # Invert class lookup dict for demo purposes
        self.class2name = {v: k for k, v in self.name2class.items()}
        # self.motion_data = list(self.data.values())
        self.motion_data = []
        for name in self.names:
            self.motion_data.append(np.array([val for val in self.data[name].values()]))

        if self.transform:
            self.scalers = [MinMaxScaler() for _ in range(len(self.motion_data))]
            for i, sample in enumerate(self.motion_data):
                self.scalers[i].fit(torch.from_numpy(sample).to(torch.float))

        self._seq_lens = [s.shape[1] for s in self.motion_data]

    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.name2class.get(self.remove_subscripts_and_extension(name), None)
        if label is None:
            raise ValueError('Label is None')
        else:
            label = torch.tensor(label)
        motion_data = torch.from_numpy(self.motion_data[idx]).to(dtype=torch.float)
        motion_data = motion_data.permute(1, 0, 2)

        if self.transform:
            motion_data = self.scalers[idx].transform(motion_data)


        if self.use_valid_joints:
            motion_data = motion_data[:, ROKOKO_VALID_JOINTS, :]
        if self.return_name:
            return motion_data, label, name
        return motion_data, label

    def __len__(self):
        return len(self.motion_data)
    
    def remove_subscripts_and_extension(self, x: str)->str:
        return re.sub(r"_\d+.*", "", x)
    
    @property
    def seq_lens(self):
        return self._seq_lens
    
def my_collate_fn(batch):
    motion_data = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    lens = [md.shape[0] for md in motion_data]
    padded = pad_sequence(motion_data, batch_first=True)
    packed = pack_padded_sequence(input=padded, lengths=lens, batch_first=True, enforce_sorted=False)
    return padded, torch.tensor(labels)
