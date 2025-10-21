import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
from typing import List, Dict, Any, Callable, Optional

from data_manipulation import stack_datapoints, data_collection

if __name__ == "__main__":

    dataset = data_collection('Pendulum-v1', num_steps=10, num_episodes=None, frame_skip=2)
    dataset = stack_datapoints(dataset, 4)
    for x in dataset:
        print(x)
