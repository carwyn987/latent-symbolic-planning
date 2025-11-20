import gymnasium as gym

def get_env(env_name="LunarLander-v3", render_mode="rgb_array"):
    env = gym.make(env_name, gravity=-4.0, render_mode=render_mode)
    return env
