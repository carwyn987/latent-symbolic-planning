import logging
import numpy as np
import gymnasium as gym
from typing import List, Dict, Any, Callable, Optional
from torch.utils.data import Dataset, DataLoader

def stack_datapoints(buf: List[Dict[str, Any]], num_stack: int) -> List[List[Dict[str, Any]]]:
    """
    Stack consecutive datapoints while excluding sequences that cross episode boundaries.

    Parameters
    ----------
    buf : list of dict
        List of datapoints, each containing keys like 'obs', 'done', etc.
    num_stack : int
        Number of consecutive datapoints to stack.

    Returns
    -------
    list of list of dict
        List of stacked sequences of length `num_stack`.
    """
    stacked_buf = []
    if len(buf) - num_stack < 0:
        logging.error("Too few datapoints for requested stack size")
        raise ValueError("Too few datapoints for requested stack size")
    for i in range(len(buf)-num_stack+1):
        if not any([x["done"] for x in buf[i:i+num_stack-1]]):
            stacked_buf.append(buf[i:i+num_stack])
    return stacked_buf


def data_collection(
    env_name: str,
    num_steps: Optional[int] = 500,
    num_episodes: Optional[int] = None,
    policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    frame_skip: Optional[int] = None,
    num_act_apply: Optional[int] = 1
) -> List[Dict[str, Any]]:
    """
    Collect transition data from a Gymnasium environment.

    Parameters
    ----------
    env_name : str
        Name of the gym environment to collect data from.
    num_steps : int, optional
        Maximum number of environment steps to collect. Default is 500.
    num_episodes : int, optional
        Number of episodes to collect before stopping. Overrides `num_steps` if reached first.
    policy : callable, optional
        Policy function mapping observation → action. Defaults to random sampling.
    frame_skip : int, optional
        Only record every Nth frame. If None, record all.

    Returns
    -------
    list of dict
        Collected transitions containing 'prior_obs', 'action', 'reward', 'obs', 'done', 'episode', 'timestep'.
    """
    env = gym.make(env_name, gravity=-2.0, render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up dataloader
    
    data = []
    episode_itr = 0
    step_itr = 0
    while True:
        done = False
        while not done:
            prior_obs = np.copy(obs)
            
            a = None
            if policy:
                a = policy(obs)
            if a is None:
                a = env.action_space.sample()

            for _ in range(num_act_apply):
                obs, reward, terminated, truncated, info = env.step(a)
                env.unwrapped.lander.angle = 0
                env.unwrapped.lander.linearVelocity = np.clip(env.unwrapped.lander.linearVelocity, a_min=-1.0, a_max=1.0)
                done = terminated or truncated
                if done:
                    break
            
            # Add to dataset
            if not frame_skip or step_itr % frame_skip == 0:
                data.append(
                    {"prior_obs": prior_obs, "action": a, "reward": reward, "obs": obs, "done": done}
                )
            
            step_itr += 1
            if num_steps and step_itr >= num_steps:
                break
        else:
            episode_itr += 1
            if num_episodes and episode_itr >= num_episodes:
                break
            obs, info = env.reset()
            continue
        break
    
    env.close()
    logging.info(f"Generated a dataset of {len(data)} (SARS) tuples")
    return data

class SARSDataset(Dataset):
    def __init__(self, dataset):
        super(SARSDataset, self).__init__()
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sars = self.dataset[index]
        if isinstance(sars, list):
            prior_obs = np.stack(tuple(x["prior_obs"] for x in sars))
            action = np.stack(tuple(x["action"] for x in sars))
            obs = np.stack(tuple(x["obs"] for x in sars))
            reward = np.stack(tuple(x["reward"] for x in sars))[:,np.newaxis]
            done = np.stack(tuple(x["done"] for x in sars)).astype(np.uint8)[:,np.newaxis]
            return {'prior_obs': prior_obs, 'action': action, 'reward': reward, 'obs': obs, 'done': done}
        else:
            return sars


def extract_trajectories(dataset):
    start = 0
    for i in range(len(dataset)):
        if isinstance(dataset[i], list):
            if any([x["done"] for x in dataset[i]]) == True:
                yield dataset[start:i+1]
                start = i+1
        else:
            if dataset[i]["done"] == True:
                yield dataset[start:i+1]
                start = i+1
