import numpy as np
import torch

from typing import Tuple, Union, List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def render(
    data: Union[torch.Tensor, np.ndarray],
    data2: Optional[Union[torch.Tensor, np.ndarray]] = None,
    connections: Optional[List[Tuple[int, int]]] = None,
    mode: str = "interactive",
    fname: Optional[str] = 'animation.gif',
    fps: int = 50
):
    if connections is None:
        connections = [
            (8, 0), (0, 1), (1, 2), (2, 3), (8, 4), (4, 5), (5, 6), (6, 7),
            (8, 12), (12, 13), (13, 14), (14, 15), (14, 16), (8, 17),
            (17, 18), (18, 19), (19, 20), (19, 21), (8, 9), (9, 10), (10, 11),
        ]

    def setup_ax(ax, d, colors):
        radius = torch.max(torch.tensor(d)).item() * 2
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius, radius])
        ax.set_aspect("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(elev=90.0, azim=-90, roll=45.)
        ax.dist = 0

        right_side_joints = {0, 1, 2, 3, 17, 18, 19, 20, 21}
        joint_colors = [colors[0] if i in right_side_joints else colors[1] for i in range(len(connections) + 1)]
        points = [ax.plot([], [], [], "o", markersize=5, color=joint_colors[i])[0] for i in range(len(connections) + 1)]
        
        # Ensure lines connecting to right side joints use the right side color
        lines = []
        for connection in connections:
            if connection[0] in right_side_joints or connection[1] in right_side_joints:
                line_color = colors[0]  # Color for right side
            else:
                line_color = colors[1]  # Color for left side
            line, = ax.plot([], [], [], c=line_color, lw=2)
            lines.append(line)

        return points, lines

    fig = plt.figure(figsize=(14, 7) if data2 is not None else (7, 7))
    ax1 = fig.add_subplot(121, projection='3d') if data2 is not None else fig.add_subplot(111, projection='3d')
    ax1.title.set_text("Skeleton 1")

    colors1 = ["red", "blue"]  # Colors for ax1
    animation_elements1 = setup_ax(ax1, data, colors1)

    if data2 is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        colors2 = ["green", "purple"]  # Colors for ax2
        animation_elements2 = setup_ax(ax2, data2, colors2)
        ax2.title.set_text("Skeleton 2")
    
    def update(frame):
        artists = []
        for points, lines in [animation_elements1, animation_elements2] if data2 is not None else [animation_elements1]:
            for i, point in enumerate(points):
                d = data if points is animation_elements1[0] else data2
                x, y, z = d[frame, i, :]
                point.set_data([x], [y])
                point.set_3d_properties([z])
                artists.append(point)

            for line, connection in zip(lines, connections):
                x_data = [d[frame, connection[0], 0], d[frame, connection[1], 0]]
                y_data = [d[frame, connection[0], 1], d[frame, connection[1], 1]]
                z_data = [d[frame, connection[0], 2], d[frame, connection[1], 2]]
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)
                artists.append(line)

        return artists

    ani = FuncAnimation(fig, update, frames=data.shape[0], blit=True, interval=1000 / fps, repeat=True)

    if mode == "interactive":
        plt.show()
    else:
        if not fname.endswith(".gif"):
            fname += ".gif"
        ani.save(fname, writer=PillowWriter(fps=25, bitrate=1800))
    plt.close()