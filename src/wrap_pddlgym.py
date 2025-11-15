import pddlgym
import imageio
from tqdm import tqdm
import numpy as np

import re

def alphanum_key(s):
    """Split into string + number chunks for natural sorting."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', s)]

def sort_tuples_alphanum(tuples_list):
    return sorted(
        tuples_list,
        key=lambda tup: alphanum_key("".join(tup))  # concatenate entire tuple
    )
    
class DummyActionSpace:
    def __init__(self, sample_fn):
        self.sample_fn = sample_fn
    
    def sample(self):
        return self.sample_fn()

class EnvWrapped():
    def __init__(self):
        self.env = pddlgym.make("PDDLEnvHanoi-v0")
        (self.obs, self.objects, self.goal), _2 = self.env.reset()
        self.addback_predicates = set()
        self.action_space = DummyActionSpace(self.sample_no_obs)
        self.setup_ohe() # sets up self.all_state_vars and self.all_actions

    def step(self, action):
        action = self.all_actions[action]
        obs, reward, done, truncated, debug_info = self.env.step(action)
        obs = self.get_ohe_from_sv(obs)
        return obs, reward, done, truncated, debug_info
    
    def reset(self):
        obs, debug_info = self.env.reset()
        self.obs = obs
        obs = self.get_ohe_from_sv(obs)
        return obs, debug_info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
        
    def get_act_from_pddl(self, act):
        for i in range(len(self.all_actions)):
            if act == self.all_actions[i]:
                return i
        raise Exception("Action not found in all_actions")
    
    def sample_no_obs(self):
        return self.sample(self.obs)
    
    def sample(self, obs):
        ohe_orig = [self.all_state_vars[i] for i in range(len(obs)) if obs[i] == 1.0]
        ohe_orig.extend(list(self.addback_predicates))
        pddl_obs = pddlgym.structs.State(literals=ohe_orig, objects=self.objects, goal=self.goal)
        return self.get_act_from_pddl(self.env.action_space.sample(pddl_obs))
    
    def get_ohe_from_sv(self, obs):
        list_of_state_vars = self.extract_state_from_raw_obs(obs)
        ohe_vec = np.zeros((len(self.all_state_vars)), dtype=np.float32)
        for i, el in enumerate(self.all_state_vars):
            if el in list_of_state_vars:
                ohe_vec[i] = 1.0
        return ohe_vec
    
    def extract_state_from_raw_obs(self, obs, setup_addbacks=False):
        list_of_state_vars = []
        for element in list(obs[0]): # obs[0] is the state
            # element is a pddlgym.structs.Literal
            predicate = str(element.predicate)
            if str(predicate) == "smaller":
                if setup_addbacks:
                    self.addback_predicates.add(element)
                continue
            # variables = element.variables # list of pddlgym.structs.TypedEntity
            list_of_state_vars.append(element)
            
        return list_of_state_vars
    
    def setup_ohe(self):
        all_state_vars = set()
        all_actions = set()

        print("\nSampling environment to get all valid predicates")
        for _ in tqdm(range(10)):
            done = False
            obs, _ = self.env.reset()
            
            while not done:
                action = self.env.action_space.sample(obs)
                obs, reward, done, truncated, debug_info = self.env.step(action)
                subset_state_vars = self.extract_state_from_raw_obs(obs, setup_addbacks=True)
                all_state_vars.update(subset_state_vars)
                all_actions.add(action)
        
        print(f"Identified the following addback predicates:\n   {self.addback_predicates}\n")
        # Sort alphanumerically to maintain consistency
        all_state_vars = sorted(list(all_state_vars), key=lambda x: str(x))
        all_actions = sorted(list(all_actions))
        self.all_actions = all_actions
        
        print(f"Found {len(all_state_vars)} states and {len(all_actions)} actions possible.")
        
        self.all_state_vars = all_state_vars
        return all_state_vars
    
    def set_ohe(self, all_state_vars):
        self.all_state_vars = all_state_vars


if __name__ == "__main__":
    env = EnvWrapped()
    init_obs, debug_info = env.reset()
    #print(env.all_state_vars)
    #print(init_obs)
    img = env.render()
    #imageio.imsave("frame1.png", img)
    obs = init_obs
    done = False
    while (init_obs == obs).all() and not done:
        action = env.sample(obs)
        obs, reward, done, truncated, debug_info = env.step(action)
        
    print(action, obs)
    img = env.render()
    #imageio.imsave("frame1.png", img)