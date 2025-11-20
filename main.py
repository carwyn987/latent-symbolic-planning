import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import Counter
from typing import List, Dict, Any, Callable, Optional

from src.data_manipulation import data_collection, \
                            SARSDataset, \
                            extract_trajectories
from src.cluster import cluster, \
                    obs_to_cluster, \
                    analyze_k_clusters, \
                    plot_clusters
from src.plotter import plot_transition_graph, \
                        plot_action_vectors
from src.eval import eval_random, \
                     eval_policy, \
                     plot_descriptive_states, \
                     show_sample_execution 


if __name__ == "__main__":

    # Setup logger
    logging.basicConfig(
        filename='logs/app.log',
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Execute an algorithm that solves the continuous Lunar Lander gymnasium environment using classical planning and low-level controllers.")
    parser.add_argument("-d", "--debug", default=0, choices={0,1,2}, required=False, type=int, help="Sets the debug logging level. Also controls the display of certain visualizations.")
    parser.add_argument("-c", "--num_clusters", default=20, required=False, type=int, help="Selects the number of clusters to generate, which will each be converted to a 'state' in PDDL.")
    parser.add_argument("-s", "--num_steps", default=5000, required=False, type=int, help="Number of steps to execute during data collection phase.")
    parser.add_argument("-m", "--clustering_method", default="kmeans", required=False, type=str, help="Clustering method.")
    parser.add_argument("-a", "--num_act_apply", default=20, required=False, type=int, help="Number of actions to apply consecutively during data collection. May help increase exploration.")
    parser.add_argument("-g", "--hardcode_start_goal_states", default=True, required=False, type=bool, help="Whether or not to hardcode the start and goal states to known values, rather than determining them from rollouts.")
    parser.add_argument("-f", "--full_replan", default=True, required=False, type=bool, help="Whether to fully replan or to correct course back onto original plan.")
    parser.add_argument("-o", "--output", default="logs", required=False, type=str, help="Directory to output metrics.")
    args = parser.parse_args()
    args.env_name = "LunarLander-v3"
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # DEBUG
    final_states = []
    goal_states = []
    start_states_save = []
    start_state_equals_goal_state = []

    # Collect data
    print("Collecting Data")
    dataset = data_collection(args.env_name, num_steps=args.num_steps, num_episodes=None, policy=None, frame_skip=None, num_act_apply=args.num_act_apply)
    sars_dataset = SARSDataset(dataset)
    obss = [x["obs"] for x in sars_dataset]
    print("Data Collected")
    
    # Cluster data
    n_clusters = args.num_clusters
    l, cluster_centers = cluster(obss, n_clusters=n_clusters, algo="kmeans", add_start=True, add_end=True)
    if args.debug >= 2:
        plot_clusters(obss, cluster_centers)
        analyze_k_clusters(obss)

    # Compute transition model
    transition_samples = []
    for traj in extract_trajectories(sars_dataset):
        # Identify states
        d_states, distances = zip(*[obs_to_cluster(x, cluster_centers) for x in traj["obs"]])

        final_states.append(traj["obs"][-1])

        # Get actions that move between states
        switch_state_indeces = [(i,i+1) for i in range(len(d_states)-1) if d_states[i] != d_states[i+1]]
        switch_sa = [(int(d_states[i]), int(d_states[j]), int(traj["action"][i])) for (i,j) in switch_state_indeces]
        transition_samples += switch_sa

    # Analyze distribution of actions (due to duplicates)
    if args.debug >= 2:
        plot_transition_graph(transition_samples)
    if args.debug >= 1:
        plot_action_vectors(cluster_centers, transition_samples)
        plot_action_vectors(cluster_centers, transition_samples, arrows=False)

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
    logging.info(f"Transition map: {transition_samples_simplified}")

    # Identify goal state, start_state (if we don't know, use curiosity to map out more sars transitions?)
    total_reward_state_idx_map = {clust: np.array([0.0]) for clust in range(len(cluster_centers))}
    start_states = []
    for traj in extract_trajectories(sars_dataset):
        d_states, distances = zip(*[obs_to_cluster(x, cluster_centers) for x in traj["obs"]])
        rewards = traj["reward"]
        start_states.append(int(d_states[0]))

        # Use episode return / final state
        return_ = np.sum(rewards)
        last_state = d_states[-1]
        total_reward_state_idx_map[d_states[-1]] += return_ # TODO: Make mean, so I can fix my -99999 below

    total_reward_state_idx_arr = np.concatenate([total_reward_state_idx_map[i] for i in range(len(cluster_centers))])
    # Mask zeroes (no data) - TODO: TASK-SPECIFIC-ASSUMPTION
    total_reward_state_idx_arr = np.where(total_reward_state_idx_arr == 0,
                                        -99999999,
                                        total_reward_state_idx_arr)
    goal_state = np.argmax(total_reward_state_idx_arr)
    goal_states.append(cluster_centers[goal_state])

    start_state_counts = Counter(start_states)
    start_state, count = start_state_counts.most_common(1)[0]
    logging.info(f"Planned start_state: {start_state}, goal_state: {goal_state}")
    if start_state == goal_state:
        logging.warning("WARNING: START STATE == GOAL STATE")
        start_state_equals_goal_state.append(1)
    else:
        start_state_equals_goal_state.append(0)
    start_states_save.append(start_state)
    
    ### TEMPORARY ####
    start_state = len(cluster_centers)-2 # obs_to_cluster([0,1.5,0,0,0,0,0,0], c)
    goal_state = len(cluster_centers)-1 # obs_to_cluster([0,0,0,0,0,0,0,0], c)

       
    #########################################################################
    # EVALUATION
    #########################################################################
    plot_descriptive_states(obss, start_states_save, goal_states, final_states, cluster_centers=cluster_centers)
  
    # Test execution for qualitative result analysis
    if args.debug > 0:
        show_sample_execution(args, cluster_centers, transition_samples_simplified, start_state, goal_state)
        
    eval_policy(args, cluster_centers, transition_samples_simplified, goal_state) # Should come first, b/c planner spits out so much content
    eval_random(args)
    
    plt.legend()
    if args.debug > 0:
        plt.show()
