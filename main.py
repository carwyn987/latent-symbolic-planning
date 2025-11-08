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
    
    returns = []
    
    cur_policy=None
    c = None
    
    for outer_loop_idx in range(3):
    
        # Collect data
        env_name = "LunarLander-v3" #'Pendulum-v1' #"CarRacing-v3" # "CartPole-v1" 
        dataset = data_collection(env_name, num_steps=10000, policy=cur_policy, frame_skip=None)
        # dataset = stack_datapoints(dataset, 4)

        dataset = SARSDataset(dataset)
        # for k,v in dataset[0].items():
        #     print(k, " : ", v.shape if isinstance(v,np.ndarray) else type(v))
            
        # Define a size threshold
        # If > threshold, we need to autoencode
        
        obss = [x["obs"] for x in dataset]
        #analyze_k_clusters(obss)
        l, c = cluster(obss, n_clusters=9, algo="kmeans")
        # plot_clusters(obss, c)
        
        transition_samples = []
        for traj in extract_trajectories(dataset):
            # Identify states
            d_states, distances = zip(*[obs_to_cluster(x, c) for x in traj["obs"]])
            
            # Get actions that move between states
            switch_state_indeces = [(i,i+1) for i in range(len(d_states)-1) if d_states[i] != d_states[i+1]]
            switch_sa = [(int(d_states[i]), int(d_states[j]), int(traj["action"][i])) for (i,j) in switch_state_indeces]
            transition_samples += switch_sa
            
        # Analyze distribution of actions
        # Since I'm using the one-step transition, we don't have a distr
        # Actually , we do, because we have duplicates
        
        #print(transition_samples)
        #plot_transition_graph(transition_samples)
        
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
        
        start_state_counts = Counter(start_states)
        start_state, count = start_state_counts.most_common(1)[0]
        print("Planned start_state: ", start_state, ", goal_state: ", goal_state)
        if start_state == goal_state:
            print("WARNING: START STATE == GOAL STATE")
            logging.warning("WARNING: START STATE == GOAL STATE")
        
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
                    break

                # Get the first step of the plan
                s_from, s_to = plan_transitions[0]
                action = state_action_map.get((s_from, s_to))
                if action is None:
                    # print(f"No known action for transition s{s_from}→s{s_to}. Sampling random action.")
                    action = env.action_space.sample()

                # print(f"Executing transition s{s_from}→s{s_to} with action {action}")

                # Execute until new cluster reached or episode ends
                #for step in range(100):
                while not done:
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
    
    
    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.show()