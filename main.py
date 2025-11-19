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
import copy

from src.data_manipulation import stack_datapoints, \
                            data_collection, \
                            SARSDataset, \
                            extract_trajectories

from src.cluster import cluster, \
                    obs_to_cluster, \
                    analyze_k_clusters, \
                    plot_clusters

from src.plotter import plot_transition_graph, \
                        plot_action_vectors
from src.planner import plan, policy
from src.eval import eval_random, \
                     eval_policy
from src.pid import *

if __name__ == "__main__":

    # Setup logger
    logging.basicConfig(
        filename='logs/app.log',
        filemode='w',
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

    # Collect data
    
    print("Collecting Data")
    env_name = "LunarLander-v3" #"Blackjack-v1" #"CliffWalking-v0"   #'Pendulum-v1' #"CarRacing-v3" # "CartPole-v1" 
    dataset = data_collection(env_name, num_steps=5000, num_episodes=None, policy=cur_policy, frame_skip=None, num_act_apply=num_act_apply)
    # dataset = stack_datapoints(dataset, 4)
    sars_dataset = SARSDataset(dataset)
    obss = [x["obs"] for x in sars_dataset]
    print("Data Collected")
    n_clusters = 20
    l, c = cluster(obss, n_clusters=n_clusters, algo="kmeans", add_start=True, add_end=True)
    # plot_clusters(obss, c)
    # analyze_k_clusters(obss)


    for outer_loop_idx in range(1):
        
        if outer_loop_idx != 0:
            print("Collecting Data")
            dataset.extend(data_collection(env_name, num_steps=5000, num_episodes=None, policy=cur_policy, frame_skip=None, num_act_apply=num_act_apply))
            sars_dataset = SARSDataset(dataset)
            obss = [x["obs"] for x in sars_dataset]
            print("Data Collected")

        transition_samples = []
        for traj in extract_trajectories(sars_dataset):
            # Identify states
            d_states, distances = zip(*[obs_to_cluster(x, c) for x in traj["obs"]])

            final_states.append(traj["obs"][-1])

            # Get actions that move between states
            switch_state_indeces = [(i,i+1) for i in range(len(d_states)-1) if d_states[i] != d_states[i+1]]
            switch_sa = [(int(d_states[i]), int(d_states[j]), int(traj["action"][i])) for (i,j) in switch_state_indeces]
            transition_samples += switch_sa

        # Analyze distribution of actions (due to duplicates)
        plot_transition_graph(transition_samples)
        plot_action_vectors(c, transition_samples)

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
        #plot_transition_graph(transition_samples_simplified)
        #plot_action_vectors(c, transition_samples_simplified)
        logging.info("TRANSITION MAP")
        logging.info(transition_samples_simplified)

        # Identify goal state, start_state (if we don't know, use curiosity to map out more sars transitions?)
        total_reward_state_idx_map = {clust: np.array([0.0]) for clust in range(len(c))}
        start_states = []
        for traj in extract_trajectories(sars_dataset):
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
        start_state = len(c)-2 # obs_to_cluster([0,1.5,0,0,0,0,0,0], c)
        goal_state = len(c)-1 # obs_to_cluster([0,0,0,0,0,0,0,0], c)

        # Planner
        plan_actions, plan_transitions, state_action_map = plan(transition_samples_simplified, start_state, goal_state, len(c))

        # --------------------------------------------------------------------------
        # 4. Execute plan in the environment (with online replanning)
        # --------------------------------------------------------------------------
        print("Executing plan in environment...")
        logging.info("EXECUTING")
        env = gym.make(env_name, gravity=-4.0, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="logs/")

        num_trajectories_to_gen = 1
        for itr in range(num_trajectories_to_gen):
            logging.info(f"Train Trajectory #{itr}")
            plan_failures.append(0)
            random_action_choices.append(0)

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
                plan_failures[-1] += 1
                continue

            print(f"Starting execution from cluster s{current_state}, goal cluster s{goal_state}")

            pidc_x = PIDController(0.01, 0.0, 0.0)
            pidc_y = PIDController(0.01, 0.0, 0.4)
            # Steps in episode
            cur_state_save = copy.copy(current_state)
            step = 0
            max_steps = 2500
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
                        print(f"No plan found from s{current_state} to s{goal_state}. Skipping.")
                        plan_failures[-1] += 1
                        continue
                    
                    pidc_x.reset()
                    pidc_y.reset()
                elif current_state != cur_state_save and current_state == s_to:
                    plan_transitions.pop(0)
                    if len(plan_transitions) == 0:
                        logging.info("Ran out of plan transitions, exiting... early exiting")
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
                    print("Episode ended early.")
                    break

            print(f"Final cluster: s{current_state}, goal: s{goal_state}, done: {done}")
            if current_state == goal_state:
                print("Goal achieved.")
            else:
                print("️Plan execution stopped before reaching goal.")

    env.close()
    cur_policy = partial(policy, state_action_map, c)
    breakpoint()

    #########################################################################
    # EVALUATION
    #########################################################################

    # Test Run
    env = gym.make(env_name, gravity=-4.0, render_mode="human")
    obs, info = env.reset()
    done = False
    
    test_cluster_list = []
    test_action_list = []

    current_state, _ = obs_to_cluster(obs, c)
    current_state = int(current_state)
    step=0
    max_steps=500
    while not done and step<max_steps:
        step+=1
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
        env.unwrapped.lander.angle = 0
        done = terminated or truncated
        current_state, _ = obs_to_cluster(obs, c)
        current_state = int(current_state)
        test_cluster_list.append(current_state)
        test_action_list.append(action)
    env.close()

    # --- FIGURE 1: EPISODE-LEVEL METRICS ---
    fig1, ax = plt.subplots(4, 1, figsize=(9, 12))

    # Returns
    ax[0].plot(returns)
    ax[0].set_xlabel("Episode", fontsize=11)
    ax[0].set_ylabel("Return", fontsize=11)
    ax[0].set_title("Episode Returns Over Evaluation", fontsize=13)

    # Plan Failures
    ax[1].plot(plan_failures)
    ax[1].set_xlabel("Episode", fontsize=11)
    ax[1].set_ylabel("Plan Failures", fontsize=11)
    ax[1].set_title("Number of Planning Failures per Episode", fontsize=13)

    # Random Actions
    ax[2].plot(random_action_choices)
    ax[2].set_xlabel("Episode", fontsize=11)
    ax[2].set_ylabel("Random Actions", fontsize=11)
    ax[2].set_title("Random Action Selections per Episode", fontsize=13)

    # Start State = Goal State
    ax[3].plot(start_state_equals_goal_state)
    ax[3].set_xlabel("Episode", fontsize=11)
    ax[3].set_ylabel("Indicator", fontsize=11)
    ax[3].set_title("Start State Equals Goal State (Binary Indicator)", fontsize=13)
    plt.tight_layout()

    # --- FIGURE 2: TEST EPISODE STEP-LEVEL DATA ---
    fig15, ax15 = plt.subplots(2, 1, figsize=(9, 8))
    # Cluster trajectory during the test episode
    ax15[0].plot(test_cluster_list)
    ax15[0].set_xlabel("Step", fontsize=11)
    ax15[0].set_ylabel("Cluster", fontsize=11)
    ax15[0].set_title("Cluster Index in Test Episode Trajectory", fontsize=13)
    # Action trajectory during the test episode
    ax15[1].plot(test_action_list)
    ax15[1].set_xlabel("Step", fontsize=11)
    ax15[1].set_ylabel("Action", fontsize=11)
    ax15[1].set_title("Action Sequence in Test Episode Trajectory", fontsize=13)
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
    
    eval_policy(env_name, c, transition_samples_simplified, goal_state, num_act_apply)
    eval_random(env_name, c, transition_samples_simplified, goal_state)
    
    plt.legend()
    plt.show()
