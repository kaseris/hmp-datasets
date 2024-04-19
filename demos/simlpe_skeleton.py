import numpy as np
import torch
import hmpdata

from typing import Tuple, Union, List, Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


dataset = hmpdata.Human36MsiMLPe(
    data_dir="datasets", split_name="train", config=hmpdata.Human36MDatasetConfig()
)
observation, target = dataset[34]
xyz_info = observation.numpy().reshape(-1, 22, 3)

def render(
    data: Union[torch.Tensor, np.ndarray],
    joint_colors: List[str] = ["red", "blue"],
    connections: Optional[List[Tuple[int, int]]] = None,
    mode: str = "interactive",
):
    """
    Render the 3D animation of the given data.

    Args:
        data: The data to render. The shape should be (T, J, 3), where T is the number of frames, J is the number of joints,
            and 3 represents the x, y, and z coordinates of each joint. Note J should be 22.
        connections: The connections between joints. If set to None, the default connections will be used.
        joint_colors: The colors of each joint.
    """
    if connections is None:
        connections = [
            (8, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (8, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (8, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (14, 16),
            (8, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (19, 21),
            (8, 9),
            (9, 10),
            (10, 11),
        ]
    assert mode in ['interactive', 'gif'], "mode should be either 'interactive' or 'gif'."

    radius = torch.max(torch.tensor(data)).item() * 2
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=90.0, azim=-90)
    ax.dist = 0

    right_side_joints = {0, 1, 2, 3, 17, 18, 19, 20, 21}
    right_side_color = "red"
    left_side_color = "black"
    joint_colors = [
        right_side_color if i in right_side_joints else left_side_color
        for i in range(len(connections) + 1)
    ]

    # Create points with distinct colors
    points = [
        ax.plot([], [], [], "o", markersize=5, c=joint_colors[i])[0]
        for i in range(len(connections) + 1)
    ]

    # Update lines to reflect right and left side connections with distinct colors
    lines = []
    for connection in connections:
        color = (
            right_side_color
            if connection[0] in right_side_joints or connection[1] in right_side_joints
            else left_side_color
        )
        (line,) = ax.plot([], [], [], c=color, lw=2)
        lines.append(line)

    # Update text annotations for each joint
    joint_texts = [ax.text(0, 0, 0, "", color="black") for _ in range(len(connections) + 1)]

    def update(frame):
        artists = []

        # Update the points and lines for each joint
        for i, point in enumerate(points):
            x, y, z = xyz_info[frame, i, :]
            point.set_data(x, y)
            point.set_3d_properties(z)
            joint_texts[i].set_position((x, y))
            joint_texts[i].set_3d_properties(z)
            joint_texts[i].set_text(str(i))
            artists.append(point)
            artists.append(joint_texts[i])

        # Update the lines connecting joints
        for line, connection in zip(lines, connections):
            x_data = [data[frame, connection[0], 0], data[frame, connection[1], 0]]
            y_data = [data[frame, connection[0], 1], data[frame, connection[1], 1]]
            z_data = [data[frame, connection[0], 2], data[frame, connection[1], 2]]
            line.set_data(x_data, y_data)
            line.set_3d_properties(z_data)
            artists.append(line)

        return artists

    ani = FuncAnimation(
    fig, update, frames=xyz_info.shape[0] - 1, blit=True, interval=25, repeat=True
    )
    if mode == "interactive":
        plt.show()
    else:
        writer = animation.PillowWriter(fps=25, bitrate=100)
        ani.save("animation.gif", writer=writer)

render(xyz_info, joint_colors=["red", "blue"], connections=None)