import torch
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

from data_manipulation import stack_datapoints, \
                            data_collection, \
                            SARSDataset, \
                            extract_trajectories

from cluster import cluster, \
                    obs_to_cluster, \
                    analyze_k_clusters, \
                    plot_clusters

from plotter import plot_transition_graph

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
    plot_transition_graph(transition_samples)