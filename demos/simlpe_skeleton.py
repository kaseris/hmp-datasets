import numpy as np
import torch
import hmpdata

from typing import Tuple
from hmpdata.human36m._simlpe.h36m_human_model import H36MHuman
from hmpdata.human36m._simlpe.misc import rotmat2xyz_torch, expmap2rotmat_torch
from hmpdata.visualization.visualization import render_animation

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


info = open('datasets/S1/directions_1.txt').readlines()
pose_info = []
for line in info:
    line = line.strip().split(',')
    if len(line) > 0:
        pose_info.append(np.array([float(x) for x in line]))
pose_info = np.array(pose_info)
T = pose_info.shape[0]
pose_info = pose_info.reshape(-1, 33, 3)
pose_info[:, :2] = 0
pose_info = pose_info[:, 1:, :].reshape(-1, 3)
pose_info = expmap2rotmat_torch(torch.tensor(pose_info).float()).reshape(T, 32, 3, 3)
xyz_info = rotmat2xyz_torch(pose_info)
# human = H36MHuman(device='cpu')
# data = human(pose_info)
print(f'data.shape: {xyz_info.shape}')
used_joints = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
xyz_info = xyz_info[:, used_joints, :]
print(xyz_info[:10, ...].max())
# render_animation(data=xyz_info.numpy(), skeleton=hmpdata.Human36MSkeleton, fps=50)

# dataset = hmpdata.Human36MsiMLPe(data_dir='datasets', split_name='train', config=hmpdata.Human36MDatasetConfig())
# observation, target = dataset[0]
# print(f'observation.shape: {observation.shape}')
# print(f'max: {observation.max()}')

# 3d scatter plot of the first frame
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=97., azim=-90, roll=0)
"""radius = torch.max(hmpdata.Human36MSkeleton.offsets()).item() * 5
ax.set_xlim3d([-radius/2, radius/2])
ax.set_zlim3d([0, radius])
ax.set_ylim3d([-radius/2, radius/2])
ax.set_aspect('equal')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.dist = 7.5
# Also add a label of index over each point
for i in range(22):
    ax.text(xyz_info[0, i, 0], xyz_info[0, i, 1], xyz_info[0, i, 2], str(i))
# Connect the points in this order
# 8 -> 0, 0->1, 1->2, 2->3
# 8 -> 4, 4->5, 5->6, 6->7
# 8 -> 12, 12->13, 13->14, 14->15, 14->16
# 8 -> 17, 17->18, 18->19, 19->20, 19->21
# 8-> 9, 9->10, 10->11
connections = [
    (8, 0), (0, 1), (1, 2), (2, 3),
    (8, 4), (4, 5), (5, 6), (6, 7),
    (8, 12), (12, 13), (13, 14), (14, 15), (14, 16),
    (8, 17), (17, 18), (18, 19), (19, 20), (19, 21),
    (8, 9), (9, 10), (10, 11)
]
# Connect hte points
for connection in connections:
    ax.plot([xyz_info[0, connection[0], 0], xyz_info[0, connection[1], 0]],
            [xyz_info[0, connection[0], 1], xyz_info[0, connection[1], 1]],
            [xyz_info[0, connection[0], 2], xyz_info[0, connection[1], 2]], c='black')
ax.scatter(xyz_info[0, :, 0], xyz_info[0, :, 1], xyz_info[0, :, 2])
plt.show()"""
dataset = hmpdata.Human36MsiMLPe(data_dir='datasets', split_name='train', config=hmpdata.Human36MDatasetConfig())
observation, target = dataset[2344]
xyz_info = observation.numpy().reshape(-1, 22, 3)
print(f'observation.shape: {xyz_info.shape}')
radius = torch.max(torch.tensor(xyz_info)).item() * 2

# Prepare the connections as you've outlined
connections = [
    (8, 0), (0, 1), (1, 2), (2, 3),
    (8, 4), (4, 5), (5, 6), (6, 7),
    (8, 12), (12, 13), (13, 14), (14, 15), (14, 16),
    (8, 17), (17, 18), (18, 19), (19, 20), (19, 21),
    (8, 9), (9, 10), (10, 11)
]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-radius/2, radius/2])
ax.set_ylim3d([-radius/2, radius/2])
ax.set_zlim3d([0, radius])
ax.set_aspect('equal')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.view_init(elev=90., azim=-90)
ax.dist = 0

lines = [ax.plot([], [], [], c='black')[0] for _ in connections]
points, = ax.plot([], [], [], 'o', markersize=5)

def update(frame):
    # Update the points
    points.set_data(xyz_info[frame, :, 0], xyz_info[frame, :, 1])
    points.set_3d_properties(xyz_info[frame, :, 2])
    
    # Update the lines connecting joints
    for line, connection in zip(lines, connections):
        x_data = [xyz_info[frame, connection[0], 0], xyz_info[frame, connection[1], 0]]
        y_data = [xyz_info[frame, connection[0], 1], xyz_info[frame, connection[1], 1]]
        z_data = [xyz_info[frame, connection[0], 2], xyz_info[frame, connection[1], 2]]
        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)

    return [points, *lines]

# Creating the animation
ani = FuncAnimation(fig, update, frames=xyz_info.shape[0] - 1, blit=True, interval=50, repeat=True)
plt.show()
# ani.save('animation.gif', writer='imagemagick', fps=50)
# writer = animation.PillowWriter(fps=50, bitrate=100)
# ani.save('animation.gif', writer=writer)