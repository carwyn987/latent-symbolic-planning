from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
import copy
import json
import os

from src.pid import PIDController
from src.cluster import obs_to_cluster
from src.planner import plan
from src.env import get_env
from src.planner import Policy


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

def eval_policy(args, cluster_centers, transition_samples_simplified, goal_state):
    returns = []
    successes = []
    steps = []

    for traj_idx in range(30):
        with open(os.path.join("logs", f"traj_log_{traj_idx}.txt"), "w") as f:
            f.write(f"cluster_centers={str([(i,[float(cluster_centers[i][0]), float(cluster_centers[i][1])]) for i in range(len(cluster_centers))])}\n")
            env = get_env(args.env_name)
            if args.debug >= 1:
                env = gym.wrappers.RecordVideo(env, video_folder="logs/", name_prefix=f"run_{traj_idx}")

            obs, _ = env.reset()
            start_state, _ = obs_to_cluster(obs, cluster_centers)
            start_state = int(start_state)

            policy = Policy(
                args=args,
                cluster_centers=cluster_centers,
                transition_samples=transition_samples_simplified,
                start_state=start_state,
                goal_state=goal_state
            )

            done = False
            _return = 0
            step = 0

            while not done:
                f.write(f"STEP {step}\n")
                step += 1
                action = policy.choose_action(obs)
                f.write(f"   current_pos={str(obs[0:2])}\n")
                f.write(f"   cluster={str(int(obs_to_cluster(obs, cluster_centers)[0]))}\n")
                f.write(f"   plan={str(policy.replan_prepend + policy.plan_transitions)}\n")
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                env.unwrapped.lander.angle = 0
                env.unwrapped.legs[0].angle = 0
                env.unwrapped.legs[1].angle = 0

                _return += reward

            current_state, _ = obs_to_cluster(obs, cluster_centers)
            current_state = int(current_state)

            returns.append(_return)
            successes.append(did_succeed(env, obs))
            steps.append(step)

            env.close()

    plot_eval_results(args, returns, successes, steps, "final_policy")

def eval_random(args):
    """
    Executes a random policy in the environment, as a baseline for our method.
    """
    env = get_env(args.env_name) 
       
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
        while not done:
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
    plot_eval_results(args, returns, successes, steps, "random_policy")


def plot_eval_results(args, returns, successes, steps, name):
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
    gen_stats(args, returns, name, "Returns", "return")
    gen_stats(args, steps, name, "Steps", "steps")

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
    fig.suptitle(f"{name}", fontsize=16)
    fig.tight_layout(h_pad=4)
    plt.title(f"{name}")

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


def gen_stats(args, scores, policy_name, title, metric_name):
    stat_dict = {
        "policy_name": policy_name,
        "num_clusters": args.num_clusters,
        "num_steps": args.num_steps,
        "clustering_method": args.clustering_method,
        "num_act_apply": args.num_act_apply,
        "hardcode_start_goal_states": args.hardcode_start_goal_states,
        "full_replan": args.full_replan,
        "title": title,
        "metric": metric_name,
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std_dev": float(np.std(scores)),
        "min": float(np.min(scores)),
        "percentile_25": float(np.percentile(scores, 25)),
        "percentile_75": float(np.percentile(scores, 75)),
        "max": float(np.max(scores)),
    }

    print(str(stat_dict))
    logging.info(str(stat_dict))
    with open(os.path.join(args.output, f"{policy_name}_{metric_name}.json"), "w") as f:
        json.dump(stat_dict, f, indent=2)
    return stat_dict


def plot_descriptive_states(obss, start_states_save, goal_states, final_states, cluster_centers):
    
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot()

    obss2 = np.array([x.flatten() for x in obss])
    ax2.scatter(
        obss2[:, 0], obss2[:, 1],
        c="gray", s=10, marker="o", alpha=0.4, label="Trajectories"
    )
    final_states = [np.array(x) for x in final_states]
    final_states_np = np.stack(final_states, axis=0)
    ax2.scatter(
        final_states_np[:, 0], final_states_np[:, 1],
        c="blue", s=30, marker="o", alpha=0.4, label="Final States"
    )

    start_states_save = [cluster_centers[x] for x in start_states_save]
    start_states_save = np.stack(start_states_save, axis=0)
    ax2.scatter(
        start_states_save[:, 0], start_states_save[:, 1],
        c="green", s=70, marker="X", alpha=0.8, label="Start State(s)"
    )

    goal_clusters = np.stack(goal_states, axis=0)
    ax2.scatter(
        goal_clusters[:, 0], goal_clusters[:, 1],
        c="red", s=70, marker="X", alpha=0.7, label="Goal Cluster(s)"
    )
    print("Goal clusters size: ", goal_clusters.shape)

    cluster_centers = np.stack(cluster_centers, axis=0)
    ax2.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c="black", s=30, marker="o", alpha=1, label="Cluster centers"
    )
    # Add labels (indices)
    for idx in range(cluster_centers.shape[0]):
        ax2.text(
            cluster_centers[idx][0], cluster_centers[idx][1],
            str(idx),
            fontsize=10,
            ha="center",
            va="center",
            color="red"
        )

    # --- Plot direction vectors ---
    ax2.quiver(
        cluster_centers[:, 0], cluster_centers[:, 1],   # start points
        cluster_centers[:, 2], cluster_centers[:, 3],   # direction vectors (u, v)
        angles='xy', scale_units='xy', scale=1, color='black', width=0.003
    )


def show_sample_execution(args, cluster_centers, transition_samples, start_state, goal_state):
    env = get_env(render_mode="human")
    policy = Policy(args, cluster_centers, transition_samples, start_state, goal_state)
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy.choose_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.unwrapped.lander.angle = 0
        env.unwrapped.legs[0].angle = 0
        env.unwrapped.legs[1].angle = 0
    env.close()