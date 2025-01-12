from typing import NamedTuple
from torch import nn
from gymnasium import Env

class PPOConfig(NamedTuple):
    env: Env
    # total time steps for training
    total_timesteps: int
    policy: str
    learning_rate: float
    # the number of steps until update during training. the agent is collecting n_steps samples, then updates, and repeating ..
    n_steps: int
    n_epochs: int
    gamma: float
    clip_range: float
    # determines how much information is printed to consule during training
    verbose: int

class ShieldConfig(NamedTuple):
    input_size: int
    num_layers: int
    buffer_size:int
    batch_size: int
    sub_batch_size:int
    learning_rate: int
    hidden_dim: int
    activation: nn.Module
    unsafe_tresh: float

class ShieldCallbackConfig(NamedTuple):
    update_freq: int
    save_freq: int
    output_dir: str