import logging
import argparse

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
from collections import Counter, defaultdict
from typing import List, Dict, Any, Callable, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from functools import partial

from src.data_manipulation import stack_datapoints, \
                            data_collection, \
                            SARSDataset, \
                            extract_trajectories

from src.cluster import cluster, \
                    obs_to_cluster, \
                    analyze_k_clusters, \
                    plot_clusters

from src.plotter import plot_transition_graph
from src.planner import plan, policy

if __name__ == "__main__":

    # Setup logger
    logging.basicConfig(
        filename='logs/app.log',
        filemode='a',  # append mode
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # DEBUG
    returns = []
    plan_failures = []
    random_action_choices = []
    final_states = []
    goal_states = []
    start_states_save = []
    start_state_equals_goal_state = []

    cur_policy=None
    c = None
    num_act_apply = 20

    for outer_loop_idx in range(3):

        # Collect data
        env_name = "LunarLander-v3" #"Blackjack-v1" #"CliffWalking-v0"   #'Pendulum-v1' #"CarRacing-v3" # "CartPole-v1" 
        dataset = data_collection(env_name, num_steps=1000, policy=cur_policy, frame_skip=None, num_act_apply=num_act_apply)
        # dataset = stack_datapoints(dataset, 4)

        dataset = SARSDataset(dataset)
        # for k,v in dataset[0].items():
        #     print(k, " : ", v.shape if isinstance(v,np.ndarray) else type(v))

        obss = [x["obs"] for x in dataset]
        # analyze_k_clusters(obss)
        l, c = cluster(obss, n_clusters=20, algo="kmeans", add_start=True, add_end=True) # 
        # plot_clusters(obss, c)

        transition_samples = []
        for traj in extract_trajectories(dataset):
            # Identify states
            d_states, distances = zip(*[obs_to_cluster(x, c) for x in traj["obs"]])

            final_states.append(traj["obs"][-1])

            # Get actions that move between states
            switch_state_indeces = [(i,i+1) for i in range(len(d_states)-1) if d_states[i] != d_states[i+1]]
            switch_sa = [(int(d_states[i]), int(d_states[j]), int(traj["action"][i])) for (i,j) in switch_state_indeces]
            transition_samples += switch_sa

        # Analyze distribution of actions (due to duplicates)
        # plot_transition_graph(transition_samples)

        # Aggregate duplicates
        transition_map = {}
        for x in transition_samples: # S, S', A
            if (x[0], x[1]) in transition_map:
                transition_map[(x[0], x[1])].append(x[2])
            else:
                transition_map[(x[0], x[1])] = [x[2]]

        # Aggregate aggregates to single action (in future, maybe prob. distr)
        for k,v in transition_map.items():
            if len(v) > 1:
                most_common_element = Counter(transition_map[k]).most_common(1)[0][0]
                transition_map[k] = most_common_element
            else:
                transition_map[k] = v[0]

        transition_samples_simplified = [(k[0], k[1], v) for k,v in transition_map.items()]
        # plot_transition_graph(transition_samples_simplified)

        # Identify goal state, start_state (if we don't know, use curiosity to map out more sars transitions?)
        total_reward_state_idx_map = {clust: np.array([0.0]) for clust in range(len(c))}
        start_states = []
        for traj in extract_trajectories(dataset):
            d_states, distances = zip(*[obs_to_cluster(x, c) for x in traj["obs"]])
            rewards = traj["reward"]
            start_states.append(int(d_states[0]))
            # Naive reward : valuation of state
            # for i in range(len(d_states)):
            #     total_reward_state_idx_map[d_states[i]] += rewards[i]

            # Use episode return / final state
            return_ = np.sum(rewards)
            last_state = d_states[-1]
            total_reward_state_idx_map[d_states[-1]] += return_ # TODO: Make mean, so I can fix my -99999 below

        total_reward_state_idx_arr = np.concatenate([total_reward_state_idx_map[i] for i in range(len(c))])
        # Mask zeroes (no data) - TODO: TASK-SPECIFIC-ASSUMPTION
        total_reward_state_idx_arr = np.where(total_reward_state_idx_arr == 0,
                                            -99999999,
                                            total_reward_state_idx_arr)
        goal_state = np.argmax(total_reward_state_idx_arr)
        goal_states.append(c[goal_state])

        start_state_counts = Counter(start_states)
        start_state, count = start_state_counts.most_common(1)[0]
        print("Planned start_state: ", start_state, ", goal_state: ", goal_state)
        if start_state == goal_state:
            print("WARNING: START STATE == GOAL STATE")
            logging.warning("WARNING: START STATE == GOAL STATE")
            start_state_equals_goal_state.append(1)
        else:
            start_state_equals_goal_state.append(0)
        start_states_save.append(start_state)
        
        ### TEMPORARY ####
        start_state = 20 # obs_to_cluster([0,1.5,0,0,0,0,0,0], c)
        goal_state = 21 # obs_to_cluster([0,0,0,0,0,0,0,0], c)
        print(start_state, goal_state)

        # Planner
        plan_actions, plan_transitions, state_action_map = plan(transition_samples_simplified, start_state, goal_state, len(c))
        # print("Plan actions: ", plan_actions, ", plan transitions: ", plan_transitions, ", state_action_map: ", state_action_map)

        # --------------------------------------------------------------------------
        # 4. Execute plan in the environment (with online replanning)
        # --------------------------------------------------------------------------
        print("Executing plan in environment...")

        env = gym.make(env_name, render_mode="rgb_array")

        num_trajectories_to_gen = 10
        for itr in range(num_trajectories_to_gen):
            plan_failures.append(0)
            random_action_choices.append(0)

            obs, info = env.reset()
            done = False

            current_state, _ = obs_to_cluster(obs, c)
            current_state = int(current_state)
            sa_sequence = []

            returns.append(0.0)

            print(f"Starting execution from cluster s{current_state}, goal cluster s{goal_state}")

            while not done:# and current_state != goal_state:

                # Plan from current cluster to goal
                plan_actions, plan_transitions, state_action_map = plan(
                    transition_samples_simplified, current_state, goal_state, len(c)
                )

                if not plan_transitions:
                    print(f"No plan found from s{current_state} to s{goal_state}. Stopping.")
                    plan_failures[-1] += 1
                    break

                # Get the first step of the plan
                s_from, s_to = plan_transitions[0]
                action = state_action_map.get((s_from, s_to))
                if action is None:
                    # print(f"No known action for transition s{s_from}→s{s_to}. Sampling random action.")
                    action = env.action_space.sample()
                    random_action_choices[-1] += 1

                # print(f"Executing transition s{s_from}→s{s_to} with action {action}")

                # Execute until new cluster reached or episode ends
                while not done:
                    for _ in range(num_act_apply):
                        obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    new_state, _ = obs_to_cluster(obs, c)
                    new_state = int(new_state)
                    returns[-1] += reward

                    # Log executed (state, action) pair
                    sa_sequence.append((current_state, action))

                    if new_state != current_state:
                        # print(f"Cluster change detected: s{current_state} → s{new_state}")
                        current_state = new_state
                        break

                    if done:
                        break

                if done:
                    print("Episode ended early.")
                    break

            print(f"Final cluster: s{current_state}, goal: s{goal_state}, done: {done}")
            if current_state == goal_state:
                print("Goal achieved.")
            else:
                print("️Plan execution stopped before reaching goal.")

            # print("Executed (state, action) sequence:")
            # for (s, a) in sa_sequence:
            #     print(f"  (s{s}, {a})")

    env.close()
    cur_policy = partial(policy, state_action_map, c)

    # EVALUATION

    # Test Run
    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset()
    done = False

    current_state, _ = obs_to_cluster(obs, c)
    current_state = int(current_state)
    while not done:
        plan_actions, plan_transitions, state_action_map = plan(
            transition_samples_simplified, current_state, goal_state, len(c)
        )
        if not plan_transitions:
            print(f"No plan found from s{current_state} to s{goal_state}. Stopping.")
            break

        # Get the first step of the plan
        s_from, s_to = plan_transitions[0]
        action = state_action_map.get((s_from, s_to))
        if action is None:
            # print(f"No known action for transition s{s_from}→s{s_to}. Sampling random action.")
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        new_state, _ = obs_to_cluster(obs, c)
        new_state = int(new_state)
    env.close()

    fig, ax = plt.subplots(4,1,figsize=(9,12))
    ax[0].plot(returns)
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Return")
    ax[1].plot(plan_failures)
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Plan Failures")
    ax[2].plot(random_action_choices)
    ax[2].set_xlabel("Episode")
    ax[2].set_ylabel("Random Actions Chosen")
    ax[3].plot(start_state_equals_goal_state)
    ax[3].set_xlabel("Episode")
    ax[3].set_ylabel("Start state == Goal state")
    plt.tight_layout()

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

    start_states_save = [c[x] for x in start_states_save]
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

    cluster_centers = np.stack(c, axis=0)
    ax2.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c="black", s=30, marker="o", alpha=1, label="Cluster centers"
    )
    # --- Plot direction vectors ---
    ax2.quiver(
        cluster_centers[:, 0], cluster_centers[:, 1],   # start points
        cluster_centers[:, 2], cluster_centers[:, 3],   # direction vectors (u, v)
        angles='xy', scale_units='xy', scale=1, color='black', width=0.003
    )
    plt.legend()

    print("Cluster centers size: ", cluster_centers.shape)

    print("Transition Samples: \n", transition_samples_simplified)

    plt.show()
