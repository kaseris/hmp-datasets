import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from hmpdata.amass.angle_to_joint import ang2joint

import torch
from torch.utils.data import Dataset


class AMASSDataset(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, actions=None, split=0, data_aug: bool = False):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = data_dir
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.skip_rate = skip_rate
        self.data_aug = data_aug
        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)
        seq_len = self.in_n + self.out_n

        amass_splits = [
            ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD'],
            ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
            ['BioMotionLab_NTroje'],
        ]
        # amass_splits = [['BioMotionLab_NTroje'], ['HumanEva'], ['SSM_synced']]
        # amass_splits = [['HumanEva'], ['HumanEva'], ['HumanEva']]
        # amass_splits[0] = list(
        #     set(amass_splits[0]).difference(set(amass_splits[1] + amass_splits[2])))

        # from human_body_prior.body_model.body_model import BodyModel
        # from smplx import lbs
        # root_path = os.path.dirname(__file__)
        # bm_path = root_path[:-6] + '/body_models/smplh/neutral/model.npz'
        # bm = BodyModel(bm_path=bm_path, num_betas=16, batch_size=1, model_type='smplh')
        # beta_mean = np.array([0.41771687, 0.25984767, 0.20500051, 0.13503872, 0.25965645, -2.10198147, -0.11915666,
        #                       -0.5498772, 0.30885323, 1.4813145, -0.60987528, 1.42565269, 2.45862726, 0.23001716,
        #                       -0.64180912, 0.30231911])
        # beta_mean = torch.from_numpy(beta_mean).unsqueeze(0).float()
        # # Add shape contribution
        # v_shaped = bm.v_template + lbs.blend_shapes(beta_mean, bm.shapedirs)
        # # Get the joints
        # # NxJx3 array
        # p3d0 = lbs.vertices2joints(bm.J_regressor, v_shaped)  # [1,52,3]
        # p3d0 = (p3d0 - p3d0[:, 0:1, :]).float().cuda().cpu().data.numpy()
        # parents = bm.kintree_table.data.numpy()[0, :]
        # np.savez_compressed('smpl_skeleton.npz', p3d0=p3d0, parents=parents)

        # load mean skeleton
        skel = np.load(os.path.join(self.path_to_data, 'body_models/smpl_skeleton.npz'))
        p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            parent[i] = parents[i]
        n = 0
        for ds in amass_splits[split]:
            if not os.path.isdir(self.path_to_data + ds):
                print(ds)
                continue
            print('>>> loading {}'.format(ds))
            for sub in os.listdir(self.path_to_data + ds):
                if not os.path.isdir(self.path_to_data + ds + '/' + sub):
                    continue
                for act in os.listdir(self.path_to_data + ds + '/' + sub):
                    if not act.endswith('.npz'):
                        continue
                    # if not ('walk' in act or 'jog' in act or 'run' in act or 'treadmill' in act):
                    #     continue
                    pose_all = np.load(self.path_to_data + ds + '/' + sub + '/' + act)
                    try:
                        poses = pose_all['poses']
                    except:
                        print('no poses at {}_{}_{}'.format(ds, sub, act))
                        continue
                    frame_rate = pose_all['mocap_framerate']
                    # gender = pose_all['gender']
                    # dmpls = pose_all['dmpls']
                    # betas = pose_all['betas']
                    # trans = pose_all['trans']
                    fn = poses.shape[0]
                    sample_rate = int(frame_rate // 25)
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float().cuda()
                    poses = poses.reshape([fn, -1, 3])
                    # remove global rotation
                    poses[:, 0] = 0
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    p3d = ang2joint(p3d0_tmp, poses, parent)
                    # self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                    self.p3d.append(p3d.cpu().data.numpy())
                    if split == 2:
                        valid_frames = np.arange(0, fn - seq_len + 1, self.skip_rate)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, self.skip_rate)

                    # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                    self.keys.append((ds, sub, act))
                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        sequence = self.p3d[key][fs][:, self.joint_used, :]  # [input_n + output_n, 18, 3]
        if self.data_aug:
            if np.random.rand() > .5:
                idx = [i for i in range(sequence.shape[0]-1, -1, -1)]
                sequence = sequence[idx]
        x = sequence[:self.in_n, :, :]
        y = sequence[self.in_n:self.in_n + self.out_n, :, :]
        next_pose = sequence[self.in_n: self.in_n + 1, :, :]

        x = torch.from_numpy(x).float().reshape(self.in_n, -1)
        y = torch.from_numpy(y).float().reshape(self.out_n, -1)
        next_pose = torch.from_numpy(next_pose).float().reshape(1, -1)
        return x, y, next_pose