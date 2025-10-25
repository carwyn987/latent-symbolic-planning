import os
import re
import json
import logging
import argparse
import subprocess
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
from collections import Counter, defaultdict
from typing import List, Dict, Any, Callable, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from data_manipulation import stack_datapoints, \
                            data_collection, \
                            SARSDataset, \
                            extract_trajectories

from cluster import cluster, \
                    obs_to_cluster, \
                    analyze_k_clusters, \
                    plot_clusters

from plotter import plot_transition_graph
from converter.pddl_convert import write_ppddl_files

if __name__ == "__main__":
    
    # Setup logger
    logging.basicConfig(
        filename='logs/app.log',
        filemode='a',  # append mode
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    ) 

    # Collect data
    env_name = "LunarLander-v3" #'Pendulum-v1' #"CarRacing-v3" # "CartPole-v1"
    dataset = data_collection(env_name, num_steps=10000, frame_skip=None)
    # dataset = stack_datapoints(dataset, 4)

    dataset = SARSDataset(dataset)
    for k,v in dataset[0].items():
        print(k, " : ", v.shape if isinstance(v,np.ndarray) else type(v))
        
    # Define a size threshold
    # If > threshold, we need to autoencode
    
    obss = [x["obs"] for x in dataset]
    #analyze_k_clusters(obss)
    l, c = cluster(obss, n_clusters=10, algo="spectral")
    #plot_clusters(obss, c)
    
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
    for x in transition_samples:
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
    print("start_state: ", start_state, ", goal_state: ", goal_state)
    if start_state == goal_state:
        print("WARNING: START STATE == GOAL STATE")
        logging.warning("WARNING: START STATE == GOAL STATE")
    
    # Planner
    
    output_dir = "ppddl_output"
    problem_out = "p01"
    domain_out = "d01"
    
    write_ppddl_files(
        transitions=transition_samples_simplified,
        num_states=len(c),
        start_state=start_state,
        goal_state=goal_state,
        output_dir=output_dir,
        problem_name=problem_out,
        domain_name=domain_out
    )
    
    # Execute
    
    # Load in pddl file
    
    domain_file = f"{output_dir}/{domain_out}.pddl"
    problem_file = f"{output_dir}/{problem_out}.pddl"
    # plan_file = f"{output_dir}/plan.out"
    plan_json_file = f"{output_dir}/{problem_out}.plan.json"

    print("Running Safe-Planner...")

    cmd = ["./safe-planner/sp", "-j", domain_file, problem_file]
    try:
        result = subprocess.run(
            cmd,
            cwd=".",
            capture_output=True,
            text=True,
            check=True,
            timeout=120
        )
        print("Safe-Planner finished successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Safe-Planner failed.")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise RuntimeError("Safe-Planner execution failed.") from e
    except FileNotFoundError:
        raise RuntimeError("Safe-Planner binary not found at ./safe-planner/sp. Check path or permissions.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Safe-Planner timed out while computing the plan.")

    if not os.path.exists(plan_json_file):
        raise FileNotFoundError(f"Expected plan file {plan_json_file} not found. Planning likely failed.")

    # --------------------------------------------------------------------------
    # 2. Parse the generated plan JSON
    # --------------------------------------------------------------------------
    print(f"Parsing plan file: {plan_json_file}")
    with open(plan_json_file, "r") as f:
        plan_data = json.load(f)

    if "main" in plan_data and "list" in plan_data["main"]:
        plan_actions = plan_data["main"]["list"]
    elif "actions" in plan_data:
        plan_actions = plan_data["actions"]
    else:
        raise ValueError("Invalid Safe-Planner JSON structure. Could not find 'main.list' or 'actions' keys.")

    print(f"Plan contains {len(plan_actions)} actions:")
    for a in plan_actions:
        print(" ", a)

    # --------------------------------------------------------------------------
    # 3. Build state→action mapping from learned transitions
    # --------------------------------------------------------------------------
    state_action_map = {(s_from, s_to): act for s_from, s_to, act in transition_samples_simplified}

    def parse_action_token(token):
        """Parse action string like 'move-s3-s1-a1' into (3, 1)."""
        m = re.match(r"move-s(\d+)-s(\d+)-a\d+", token)
        return (int(m.group(1)), int(m.group(2))) if m else None

    plan_transitions = [parse_action_token(a) for a in plan_actions if parse_action_token(a)]

    # --------------------------------------------------------------------------
    # 4. Execute plan in the environment
    # --------------------------------------------------------------------------
    print("Executing plan in environment...")

    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset()
    done = False

    current_state, _ = obs_to_cluster(obs, c)
    current_state = int(current_state)

    for (s_from, s_to) in plan_transitions:
        if done:
            break
        if current_state != s_from:
            print(f"Expected to be in cluster s{s_from}, but in s{current_state}. Aborting.")
            break

        action = state_action_map.get((s_from, s_to))
        if action is None:
            print(f"No known action for transition s{s_from}→s{s_to}.")
            break

        # Execute until new cluster reached or episode ends
        for step in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            new_state, _ = obs_to_cluster(obs, c)
            new_state = int(new_state)
            if new_state == s_to:
                print(f"Transition s{s_from}→s{s_to} successful.")
                current_state = new_state
                break

        if done:
            print("Episode ended early.")
            break

    env.close()
    print(f"Final cluster: s{current_state}, goal: s{goal_state}")
    if current_state == goal_state:
        print("Goal achieved.")
    else:
        print("Plan execution stopped before reaching goal.")