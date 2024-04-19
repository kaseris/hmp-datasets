import numpy as np
import torch
import hmpdata

from typing import Tuple, Union, List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


dataset = hmpdata.Human36MsiMLPe(
    data_dir="datasets", split_name="train", config=hmpdata.Human36MDatasetConfig()
)
observation, target = dataset[8786]
xyz_info = observation.numpy().reshape(-1, 22, 3)

hmpdata.render_siMLPe(xyz_info, connections=None, data2=xyz_info)