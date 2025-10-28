import pytest
import numpy as np
import gymnasium as gym
from src.data_manipulation import stack_datapoints, data_collection


def test_stack_datapoints_basic():
    buf = [{"obs": i, "done": False} for i in range(5)]
    stacked = stack_datapoints(buf, num_stack=3)
    print(buf, stacked)
    assert len(stacked) == 3
    # Each stack should have 3 items and be contiguous
    for i, stack in enumerate(stacked):
        assert [x["obs"] for x in stack] == [i, i+1, i+2]


def test_stack_datapoints_with_done():
    buf = [
        {"obs": 0, "done": False},
        {"obs": 1, "done": True},
        {"obs": 2, "done": False},
        {"obs": 3, "done": False},
    ]
    stacked = stack_datapoints(buf, num_stack=2)
    print(stacked)
    # Should skip the segment crossing the 'done' flag
    # But, final sample can be 'done'
    assert all(not any(x["done"] for x in s[:-1]) for s in stacked)
    assert len(stacked) == 2


def test_data_collection_random_policy():
    data = data_collection("CartPole-v1", num_steps=20)
    assert isinstance(data, list)
    assert all(isinstance(d, dict) for d in data)
    keys = {"prior_obs", "action", "reward", "obs", "done"}
    assert keys.issubset(data[0].keys())
    assert len(data) <= 20

