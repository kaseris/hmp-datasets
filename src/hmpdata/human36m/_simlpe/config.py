from dataclasses import dataclass

@dataclass
class Human36MDatasetConfig:
    data_aug: bool = True
    h36m_input_length: int = 50
    h36m_target_length_train: int = 10
    h36m_target_length_eval: int = 25
    dim: int = 66
    shift_step: int = 1