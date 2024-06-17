import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from typing import Union, List

def plot_skeleton(skeleton: Union[List, np.ndarray],
                  connections: List,
                  ignore_joints: List[int] = None,
                  title: str = "Skeleton Plot",
                  show_names: bool = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for idx, loc in enumerate(skeleton):
        if idx in ignore_joints and ignore_joints is not None:
            continue
        ax.scatter(loc[0], loc[1], loc[2], c="black")
        if show_names:
            ax.text(loc[0], loc[1], loc[2], str(idx), color="black")
        ax.set_aspect("equal")

    # Plot bone connections
    for connection in connections:
        idx1, idx2 = connection
        loc1 = skeleton[idx1]
        loc2 = skeleton[idx2]
        ax.plot(
            [loc1[0], loc2[0]],
            [loc1[1], loc2[1]],
            [loc1[2], loc2[2]],
            "r-"
        )
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title_ = title if title is not None else "Skeleton Plot"
    ax.set_title(title_)
    plt.show()


def render_animation(sequence: np.ndarray,
                     connections: List,
                     ignore_joints: List[int] = None,
                     interval: int = 100,
                     show_names: bool = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        skeleton = sequence[frame]
        for idx, loc in enumerate(skeleton):
            if ignore_joints and idx in ignore_joints:
                continue
            ax.scatter(loc[0], loc[1], loc[2], c="black")
            if show_names:
                ax.text(loc[0], loc[1], loc[2], str(idx), color="black")
            ax.set_aspect("equal")
        for connection in connections:
            idx1, idx2 = connection
            loc1 = skeleton[idx1]
            loc2 = skeleton[idx2]
            ax.plot(
                [loc1[0], loc2[0]],
                [loc1[1], loc2[1]],
                [loc1[2], loc2[2]],
                "r-"
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Frame {frame}")

    ani = FuncAnimation(fig, update, frames=len(sequence), interval=interval)
    plt.show()