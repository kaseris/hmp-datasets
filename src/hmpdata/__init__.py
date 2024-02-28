__version__ = '1.0.0'

from hmpdata.human36m._h36m import skeleton_h36m as Human36MSkeleton
from hmpdata.misc.mocap_dataset import MocapDataset

Human36MDataset = MocapDataset