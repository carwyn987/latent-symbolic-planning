from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from src.cluster import obs_to_cluster
from src.planner import plan, policy

def did_succeed(env, obs):
    pos_succ = False
    if abs(env.unwrapped.lander.position[0]) < 0.5 and abs(env.unwrapped.lander.position[1]) < 0.5:
        pos_succ = True
    vel_succ = False
    if abs(env.unwrapped.lander.linearVelocity[0]) < 0.5 and abs(env.unwrapped.lander.linearVelocity[1]) < 0.5:
        vel_succ = True
    legs_down = False
    if obs[-1] == 1 and obs[-2] == 1:
        legs_down = True
        
    return (pos_succ, vel_succ, legs_down)

def eval_policy(env_name, c, transition_samples_simplified, goal_state, num_act_apply):
    # Test Run
    env = gym.make(env_name, gravity=-2.0, render_mode="rgb_array")
    
    num_epochs = 30
    max_steps = 500
    returns = []
    successes = []
    steps = []

    for epoch in tqdm(range(num_epochs)):
        obs, info = env.reset()
        current_state, _ = obs_to_cluster(obs, c)
        current_state = int(current_state)
        
        done = False
        step = 0
        _return = 0
        while not done and step < max_steps:
            
            action = env.action_space.sample()
            
            plan_actions, plan_transitions, state_action_map = plan(
                transition_samples_simplified, current_state, goal_state, len(c)
            )

            s_from, s_to = plan_transitions[0]
            action = state_action_map.get((s_from, s_to))
            if action is None:
                action = env.action_space.sample()
            
            for _ in range(num_act_apply):
                step += 1
                obs, reward, terminated, truncated, info = env.step(action)
                _return += reward
                env.unwrapped.lander.angle = 0
                done = terminated or truncated
        
        returns.append(_return)
        successes.append(did_succeed(env, obs))
        steps.append(step)
        
    env.close()
    plot_eval_results(returns, successes, steps, "FINAL POLICY")

def eval_random(env_name, c, transition_samples_simplified, goal_state):
    # Test Run
    env = gym.make(env_name, gravity=-2.0, render_mode="rgb_array")
    
    num_epochs = 100
    max_steps = 500
    returns = []
    successes = []
    steps = []

    for epoch in tqdm(range(num_epochs)):
        obs, info = env.reset()
        done = False
        step = 0
        _return = 0
        while not done and step < max_steps:
            step += 1
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            _return += reward
            env.unwrapped.lander.angle = 0
            done = terminated or truncated
        
        returns.append(_return)
        successes.append(did_succeed(env, obs))
        steps.append(step)
        
    env.close()
    plot_eval_results(returns, successes, steps, "RANDOM POLICY")



def eval_random(env_name, c, transition_samples_simplified, goal_state):
    # Test Run
    env = gym.make(env_name, gravity=-1.0, render_mode="rgb_array")
    
    num_epochs = 30
    max_steps = 1000
    returns = []
    successes = []
    steps = []

    for epoch in tqdm(range(num_epochs)):
        obs, info = env.reset()
        done = False
        step = 0
        _return = 0
        while not done and step < max_steps:
            step += 1
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            _return += reward
            env.unwrapped.lander.angle = 0
            done = terminated or truncated
        
        
        returns.append(_return)
        successes.append(did_succeed(env, obs))
        steps.append(step)
        
    env.close()
    plot_eval_results(returns, successes, steps, "RANDOM POLICY")


def plot_eval_results(returns, successes, steps, name):
    """
    Plot returns, steps, cumulative success rate, and print summary statistics.
    """

    returns = np.array(returns)
    s_pos, s_vel, s_legs = zip(*successes)
    s_pos = list(s_pos)
    s_vel = list(s_vel)
    s_legs = list(s_legs)
    s_all = [a and b and c for a,b,c in successes]
    steps = np.array(steps)

    # --- PRINT SUMMARY STATISTICS ---
    print(f"\n===== {name} =====")
    print("\n===== Evaluation Summary =====")
    print(f"Total episodes: {len(returns)}")

    # Returns statistics
    print("\n--- Returns ---")
    print(f"Mean return:        {np.mean(returns):.3f}")
    print(f"Median return:      {np.median(returns):.3f}")
    print(f"Std dev:            {np.std(returns):.3f}")
    print(f"25th percentile:    {np.percentile(returns, 25):.3f}")
    print(f"75th percentile:    {np.percentile(returns, 75):.3f}")
    print(f"Best return:        {np.max(returns):.3f}")
    print(f"Worst return:       {np.min(returns):.3f}")

    # Steps statistics
    print("\n--- Steps per Episode ---")
    print(f"Mean steps:         {np.mean(steps):.2f}")
    print(f"Median steps:       {np.median(steps):.2f}")
    print(f"Std dev:            {np.std(steps):.2f}")
    print(f"25th percentile:    {np.percentile(steps, 25):.2f}")
    print(f"75th percentile:    {np.percentile(steps, 75):.2f}")

    # Success rate
    pos_success_rate = np.mean(s_pos)
    vel_success_rate = np.mean(s_vel)
    legs_success_rate = np.mean(s_legs)
    all_success_rate = np.mean(s_legs)
    print(f"\nPos Success rate:       {100 * pos_success_rate:.1f}% ({np.sum(s_pos)}/{len(successes)})")
    print(f"\nVel Success rate:       {100 * vel_success_rate:.1f}% ({np.sum(s_vel)}/{len(successes)})")
    print(f"\nLegs Success rate:     {100 * legs_success_rate:.1f}% ({np.sum(s_legs)}/{len(successes)})")
    print(f"\nAll Success rate:       {100 * all_success_rate:.1f}% ({np.sum(s_all)}/{len(successes)})")

    print("================================\n")

    # --- PLOTTING ---
    epochs = np.arange(1, len(returns) + 1)
    cumulative_pos_success_rate = np.cumsum(s_pos) / epochs

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.tight_layout(h_pad=4)

    # Plot 1: Returns
    axs[0].plot(epochs, returns, linewidth=2)
    axs[0].set_title("Episode Returns")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Return")

    # Plot 2: Steps
    axs[1].plot(epochs, steps, linewidth=2)
    axs[1].set_title("Episode Length (Steps)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Steps")

    # Plot 3: Cumulative Success Rate
    axs[2].plot(epochs, cumulative_pos_success_rate, linewidth=2)
    axs[2].set_title("Cumulative Position Success Rate")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Success Rate")
    axs[2].set_ylim(0, 1)

    # plt.show()
