import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from os import PathLike
from typing import Union, List, Optional

from hmpdata import Human36MSkeleton
from hmpdata.misc.quaternion import qfix, qeuler_np


@dataclass
class Human36MCustomDatasetSample:
    """
    Args:
    ----
    - subject (str): Subject identifier.
    - action (str): Action identifier.
    - trajectory (np.ndarray): Trajectory data.
    - rotations (np.ndarray): Rotations data.
    - positions_world (np.ndarray): World positions data.
    - positions_local (np.ndarray): Local positions data.
    - rotations_euler (np.ndarray): Euler angles data.
    """
    subject: str
    action: str
    trajectory: np.ndarray
    rotations: np.ndarray
    positions_world: np.ndarray
    positions_local: np.ndarray
    rotations_euler: Optional[np.ndarray] = None

@dataclass
class Batch:
    """
    Args:
    ----
    - subjects (List[str]): List of subject identifiers.
    - actions (List[str]): List of action identifiers.
    - trajectories (torch.Tensor): Trajectories data.
    - rotations (torch.Tensor): Rotations data.
    - positions_world (torch.Tensor): World positions data.
    - positions_local (torch.Tensor): Local positions data.
    - rotations_euler (torch.Tensor): Euler angles data.
    - trajectories_labels (torch.Tensor): Trajectories labels.
    - rotations_labels (torch.Tensor): Rotations labels.
    - positions_world_labels (torch.Tensor): World positions labels.
    - positions_local_labels (torch.Tensor): Local positions labels.
    """
    subjects: List[str]
    actions: List[str]
    trajectories: torch.Tensor
    rotations: torch.Tensor
    positions_world: torch.Tensor
    positions_local: torch.Tensor
    rotations_euler: torch.Tensor
    trajectories_labels: torch.Tensor
    rotations_labels: torch.Tensor
    positions_world_labels: torch.Tensor
    positions_local_labels: torch.Tensor


class Human36MCollateFunction:
    """
    Collate function for the Human36MCustomDataset.

    Args:
    ----
    - prefix_len (int): Length of the prefix to be extracted from each sequence.
    - label_method (str): Label method to be used. Can be one of 'next', 'block'. Default is 'next'. Determines how the sequences will be labeled. If 'next', the next frame will be the label. If 'block', the next 10 frames will be the label.
    """
    def __init__(self, prefix_len: int = 25, label_method: str = 'next', target_len: int = 100) -> None:
        self._prefix_len = prefix_len
        self._label_method = label_method
        self._target_len = target_len

    def __call__(self, batch: List[Human36MCustomDatasetSample]) -> dict:
        start_indices = [torch.randint(0, len(sample.positions_world) - self._prefix_len, (1,)).item() for sample in batch]
        start_indices = torch.tensor(start_indices)
        # Slice the data
        trajectories = [torch.from_numpy(sample.trajectory[start_idx:start_idx + self._prefix_len]).unsqueeze(0) for sample, start_idx in zip(batch, start_indices)]
        rotations = [torch.from_numpy(sample.rotations[start_idx:start_idx + self._prefix_len]).unsqueeze(0) for sample, start_idx in zip(batch, start_indices)]
        positions_world = [torch.from_numpy(sample.positions_world[start_idx:start_idx + self._prefix_len]).unsqueeze(0) for sample, start_idx in zip(batch, start_indices)]
        positions_local = [torch.from_numpy(sample.positions_local[start_idx:start_idx + self._prefix_len]).unsqueeze(0) for sample, start_idx in zip(batch, start_indices)]
        # Create the same tensors for the labels, now shifted by 1
        if self._label_method == 'next':
            trajectories_labels = [torch.from_numpy(sample.trajectory[start_idx + 1:start_idx + self._prefix_len + 1]).unsqueeze(0) for sample, start_idx in zip(batch, start_indices)]
            rotations_labels = [torch.from_numpy(sample.rotations[start_idx + 1:start_idx + self._prefix_len + 1]).unsqueeze(0) for sample, start_idx in zip(batch, start_indices)]
            positions_world_labels = [torch.from_numpy(sample.positions_world[start_idx + 1:start_idx + self._prefix_len + 1]).unsqueeze(0) for sample, start_idx in zip(batch, start_indices)]
            positions_local_labels = [torch.from_numpy(sample.positions_local[start_idx + 1:start_idx + self._prefix_len + 1]).unsqueeze(0) for sample, start_idx in zip(batch, start_indices)]
        # Concatenate the data
        trajectories = torch.cat(trajectories, dim=0)
        rotations = torch.cat(rotations, dim=0)
        positions_world = torch.cat(positions_world, dim=0)
        positions_local = torch.cat(positions_local, dim=0)
        return Batch(
            subjects=[sample.subject for sample in batch],
            actions=[sample.action for sample in batch],
            trajectories=trajectories,
            rotations=rotations,
            positions_world=positions_world,
            positions_local=positions_local,
            rotations_euler=None,
            trajectories_labels=trajectories_labels,
            rotations_labels=rotations_labels,
            positions_world_labels=positions_world_labels,
            positions_local_labels=positions_local_labels
        )
        

class Human36MCustomDataset(Dataset):
    """
    Args:
    ----
    - path (str or PathLike): Path to the dataset file.
    """
    def __init__(self, path: Union[str, PathLike],
                 skeleton,
                 fps: int = 50,
                 use_gpu: bool = False,
                 subjects: Optional[List[str]] = ['S1', 'S7', 'S8', 'S9', 'S11'],
                 ) -> None:
        super().__init__()
        self.path = path
        self._skeleton = skeleton
        self._fps = fps
        self._use_gpu = use_gpu
        self._subjects = subjects
        # self.subjects = subjects
        # This will return a dictionary with the following structure:
        # {
        #     'subject': {
        #         'action': {
        #             'rotations': np.ndarray,
        #             'trajectory': np.ndarray
        #         }
        #     }
        # }
        # Therefore, we will need later to flatten it for quick access from the getitem method.
        self._data = self._load()
        self._is_flat = False
        self._flattened_data = None

    def flatten(self):
        """
        Flattens the nested dictionary structure into a list of tuples for easy indexing.
        Each tuple contains (subject, action, data_point), where data_point is a dictionary.
        """
        self._flatten_data()

    def _flatten_data(self):
        """
        Flattens the nested dictionary structure into a list of tuples for easy indexing.
        Each tuple contains (subject, action, data_point), where data_point is a dictionary.
        """
        flattened = []
        for subject, actions in self._data.items():
            for action, data_point in actions.items():
                flattened.append((subject, action, data_point))
        self._flattened_data = flattened
        self._is_flat = True

    def __len__(self):
        return len(self._flattened_data)

    def __getitem__(self, index):
        if self.is_flat:
            subject, action, data_point = self._flattened_data[index]
            sample = Human36MCustomDatasetSample(
                subject=subject,
                action=action,
                trajectory=data_point['trajectory'],
                rotations=data_point['rotations'],
                positions_world=data_point['positions_world'],
                positions_local=data_point['positions_local'],
            )
            return sample
        else:
            raise ValueError('The dataset is not flat. Please call flatten() before accessing data.')

    def _load(self):
        result = {}
        data = np.load(self.path, allow_pickle=True)
        for i, (trajectory, rotations, subject, action) in enumerate(zip(data['trajectories'],
                                                                         data['rotations'],
                                                                         data['subjects'],
                                                                         data['actions'])):
            if subject not in self._subjects:
                continue
            if subject not in result:
                result[subject] = {}
            
            result[subject][action] = {
                'rotations': rotations,
                'trajectory': trajectory
            }
        return result

    def downsample(self, factor, keep_strides=True):
        """
        Downsample this dataset by an integer factor, keeping all strides of the data
        if keep_strides is True.
        The frame rate must be divisible by the given factor.
        The sequences will be replaced by their downsampled versions, whose actions
        will have '_d0', ... '_dn' appended to their names.
        """
        assert self._fps % factor == 0
        
        for subject in self._data.keys():
            new_actions = {}
            for action in list(self._data[subject].keys()):
                for idx in range(factor):
                    tup = {}
                    for k in self._data[subject][action].keys():
                        tup[k] = self._data[subject][action][k][idx::factor]
                    new_actions[action + '_d' + str(idx)] = tup
                    if not keep_strides:
                        break
            self._data[subject] = new_actions
            
        self._fps //= factor
        
    def _mirror_sequence(self, sequence):
        mirrored_rotations = sequence['rotations'].copy()
        mirrored_trajectory = sequence['trajectory'].copy()
        
        joints_left = self._skeleton.joints_left()
        joints_right = self._skeleton.joints_right()
        
        # Flip left/right joints
        mirrored_rotations[:, joints_left] = sequence['rotations'][:, joints_right]
        mirrored_rotations[:, joints_right] = sequence['rotations'][:, joints_left]
        
        mirrored_rotations[:, :, [2, 3]] *= -1
        mirrored_trajectory[:, 0] *= -1

        return {
            'rotations': qfix(mirrored_rotations),
            'trajectory': mirrored_trajectory
        }
    
    def mirror(self):
        """
        Perform data augmentation by mirroring every sequence in the dataset.
        The mirrored sequences will have '_m' appended to the action name.
        """
        for subject in self._data.keys():
            for action in list(self._data[subject].keys()):
                if '_m' in action:
                    continue
                self._data[subject][action + '_m'] = self._mirror_sequence(self._data[subject][action])

    def compute_euler_angles(self, order):
        for subject in self._data.values():
            for action in subject.values():
                action['rotations_euler'] = qeuler_np(action['rotations'], order, use_gpu=self._use_gpu)
                
    def compute_positions(self):
        for subject in self._data.values():
            for action in subject.values():
                rotations = torch.from_numpy(action['rotations'].astype('float32')).unsqueeze(0)
                trajectory = torch.from_numpy(action['trajectory'].astype('float32')).unsqueeze(0)
                if self._use_gpu:
                    rotations = rotations.to('cuda' if torch.cuda.is_available() else 'cpu')
                    trajectory = trajectory.to('cuda' if torch.cuda.is_available() else 'cpu')
                action['positions_world'] = self._skeleton.forward_kinematics(rotations, trajectory).squeeze(0).cpu().numpy()
                
                # Absolute translations across the XY plane are removed here
                trajectory[:, :, [0, 2]] = 0
                action['positions_local'] = self._skeleton.forward_kinematics(rotations, trajectory).squeeze(0).cpu().numpy()

    @property
    def data(self):
        return self._data
    
    @property
    def is_flat(self):
        return self._is_flat
    
def build_dataset(path_to_dataset: Union[str, PathLike],
                  skeleton,
                  fps: int = 50,
                  use_gpu: bool = False,
                  mode: str = 'train',
                  augmentations = ['downsample', 'mirror'],
                  prefix_len: int = 25):
    subjects = []
    if mode == 'train':
        subjects = ['S1', 'S7', 'S8', 'S9', 'S11']
    elif mode == 'val':
        subjects = ['S6']
    elif mode == 'test':
        subjects = ['S5']
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'val', or 'test'.")

    dataset = Human36MCustomDataset(path=path_to_dataset, skeleton=skeleton, subjects=subjects, use_gpu=use_gpu, fps=fps)
    for aug in augmentations:
        if aug == 'downsample':
            dataset.downsample(2)
        elif aug == 'mirror':
            dataset.mirror()
    dataset.compute_positions()
    dataset.flatten()
    # Create the data loader
    data_loader = DataLoader(dataset, batch_size=32, collate_fn=Human36MCollateFunction(prefix_len=prefix_len))
    return dataset, data_loader


if __name__ == '__main__':
    train_set, train_loader = build_dataset(path_to_dataset='datasets/dataset_h36m.npz', skeleton=Human36MSkeleton, mode='train')
    sample = train_set[0]
    d = sample.positions_world
    from hmpdata.visualization.visualization import render_animation
    render_animation(data=d, skeleton=Human36MSkeleton, fps=50)