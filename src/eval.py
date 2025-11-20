from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
import copy

from src.pid import PIDController
from src.cluster import obs_to_cluster
from src.planner import plan
from src.env import get_env

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
    
    env = get_env(env_name)
    
    num_epochs = 30
    max_steps = 500
    returns = []
    successes = []
    steps = []

    num_trajectories_to_gen = 30
    for itr in range(num_trajectories_to_gen):
        logging.info(f"Train Trajectory #{itr}")

        obs, info = env.reset()
        done = False

        current_state, _ = obs_to_cluster(obs, c)
        current_state = int(current_state)
        sa_sequence = []

        returns.append(0.0)
        
        # Plan from current cluster to goal
        plan_actions, plan_transitions, state_action_map = plan(
            transition_samples_simplified, current_state, goal_state, len(c)
        )
        saved_plan_actions = copy.copy(plan_actions)
        logging.info(f"   Original Plan Actions: {plan_actions}")

        if not plan_transitions:
            print(f"No plan found from s{current_state} to s{goal_state}. Skipping.")
            continue

        print(f"Starting execution from cluster s{current_state}, goal cluster s{goal_state}")

        pidc_x = PIDController(0.01, 0.0, 0.0)
        pidc_y = PIDController(0.01, 0.0, 0.4)
        # Steps in episode
        cur_state_save = copy.copy(current_state)
        step = 0
        max_steps = 9999999999 # 2500
        while not done and step < max_steps:
            step += 1
            if step == max_steps:
                logging.info(f"Steps hit max_steps ... exiting")
            
            # Get the first step of the plan
            s_from, s_to = plan_transitions[0]
            
            if current_state != cur_state_save and current_state != s_to:
                full_replan = True
                if full_replan:
                    plan_actions, plan_transitions, state_action_map = plan(
                        transition_samples_simplified, current_state, goal_state, len(c)
                    )
                else:
                    pass
                logging.info(f"   failed to move to {s_to}, moved to {current_state} instead.")
                logging.info(f"   new plan: {plan_actions}")

                if not plan_transitions:
                    logging.error(f"No plan found from s{current_state} to s{goal_state}. Skipping.")
                    continue
                
                pidc_x.reset()
                pidc_y.reset()
            elif current_state != cur_state_save and current_state == s_to:
                plan_transitions.pop(0)
                if len(plan_transitions) == 0:
                    logging.error("Ran out of plan transitions, exiting... early exiting")
                    break
                s_from, s_to = plan_transitions[0]
                logging.info(f"   successfully moved to {s_to}")
                pidc_x.reset()
                pidc_y.reset()

            ######################### PID (P) LOOP LEARNER ##############################
            def choose_act_pid(s_to_clust_center, obs, pidc_x, pidc_y):
                pidc_x.set_target(s_to_clust_center[0])
                pidc_y.set_target(s_to_clust_center[1])
                move_x = pidc_x.update(obs[0])
                move_y = 5 * pidc_y.update(obs[1])
                logits = np.array([abs(move_x), abs(move_y)])
                #act_probs = np.exp(logits) / np.sum(np.exp(logits))
                act_probs = logits / np.sum(logits)
                act_idx = np.random.choice(len(act_probs), p=act_probs)
                logging.info(f"      currently at {obs[0:4]}, want to be at {s_to_clust_center[0:4]}, logits = {logits}, choose on {act_idx}, probs: {act_probs}")
                
                if act_idx == 0: # x error
                    return 3 if move_x > 0 else 1
                if act_idx == 1: # y error
                    return 2 if move_y > 0 else 0
            
            s_to_clust_center = c[s_to]
            action = choose_act_pid(s_to_clust_center, obs, pidc_x, pidc_y)
            logging.info(f"      Choosing action {action}")
            #action = env.action_space.sample()
            #########################################################################

            cur_state_save = int(obs_to_cluster(obs, c)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            env.unwrapped.lander.angle = 0
            env.unwrapped.legs[0].angle = 0
            env.unwrapped.legs[1].angle = 0
            
            done = terminated or truncated
            current_state = int(obs_to_cluster(obs, c)[0])
            returns[-1] += reward

            # Log executed (state, action) pair
            sa_sequence.append((current_state, action))

            if done:
                logging.info("Episode ended early.")
                break

        logging.info(f"Final cluster: s{current_state}, goal: s{goal_state}, done: {done}")
        if current_state == goal_state:
            print("Goal achieved.")
        else:
            print("️Plan execution stopped before reaching goal.")
    
        successes.append(did_succeed(env, obs))
        steps.append(step)
        
    env.close()
    plot_eval_results(returns, successes, steps, "FINAL POLICY")

def eval_random(env_name, c, transition_samples_simplified, goal_state):
    # Test Run
    env = get_env(env_name) 
       
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
    gen_stats(returns, "\n--- Returns ---", "return")
    gen_stats(steps, "\n--- Steps ---", "steps")

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


def gen_stats(scores, title, metric_name):
    stat_str = f"""
    \n--- {title} ---")
    Mean {metric_name}:         {np.mean(scores):.2f}
    Median {metric_name}:       {np.median(scores):.2f}
    Std dev:            {np.std(scores):.2f}
    Min:                {np.min(scores):.2f}
    25th percentile:    {np.percentile(scores, 25):.2f}
    75th percentile:    {np.percentile(scores, 75):.2f}
    Max:                {np.max(scores):.2f}\n
    """
    
    print(stat_str)
    logging.info(stat_str)
    return stat_str


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
  