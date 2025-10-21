import numpy as np
import gymnasium as gym
from typing import List, Dict, Any, Callable, Optional

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
    assert len(buf) - num_stack >= 0
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
    env = gym.make(env_name, render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up dataloader
    
    data = []
    episode_itr = 0
    step_itr = 0
    while True:
        done = False
        while not done:
            prior_obs = np.copy(obs)
            
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            
            # Add to dataset
            if not frame_skip or step_itr % frame_skip == 0:
                data.append(
                    {"prior_obs": prior_obs, "action": a, "reward": reward, "obs": obs, "done": done}
                )
            
            step_itr += 1
            if num_steps and step_itr == num_steps:
                break
        else:
            episode_itr += 1
            if num_episodes and episode_itr == num_episodes:
                break
            obs, info = env.reset()
            continue
        break
    
    env.close()
    return data