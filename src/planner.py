import subprocess
import json, os
import re
import numpy as np
import logging

from src.pddl_convert import write_ppddl_files
from src.cluster import obs_to_cluster
from src.pid import PIDController
                        
def plan(transitions, start_state, goal_state, num_states):
    
    # Create plan files
    output_dir = "ppddl_output"
    problem_out = "p01"
    domain_out = "d01"
    
    print("##############################################\nRUNNING PLANNER\n")
    write_ppddl_files(
        transitions=transitions,
        num_states=num_states,
        start_state=start_state,
        goal_state=goal_state,
        output_dir=output_dir,
        problem_name=problem_out,
        domain_name=domain_out
    )
    
    domain_file = f"{output_dir}/{domain_out}.pddl"
    problem_file = f"{output_dir}/{problem_out}.pddl"
    
    # print("Running Safe-Planner...")
    cmd = ["./safe-planner/sp", "-v 2", "-j", domain_file, problem_file]
    try:
        result = subprocess.run(
            cmd,
            cwd=".",
            capture_output=False,
            text=False,
            check=True,
            timeout=120,
            stdout=None
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
    
    plan_json_file = f"{output_dir}/{problem_out}.plan.json"
    # plan_file = f"{output_dir}/plan.out"

    if not os.path.exists(plan_json_file):
        raise FileNotFoundError(f"Expected plan file {plan_json_file} not found. Planning likely failed.")

    # Parse the generated plan JSON
    # print(f"Parsing plan file: {plan_json_file}")
    with open(plan_json_file, "r") as f:
        plan_data = json.load(f)

    if "main" in plan_data and "list" in plan_data["main"]:
        plan_actions = plan_data["main"]["list"]
    elif "actions" in plan_data:
        plan_actions = plan_data["actions"]
    else:
        raise ValueError("Invalid Safe-Planner JSON structure. Could not find 'main.list' or 'actions' keys.")

    # --------------------------------------------------------------------------
    # 3. Build state→action mapping from learned transitions
    # --------------------------------------------------------------------------
    state_action_map = {(s_from, s_to): act for s_from, s_to, act in transitions}

    def parse_action_token(token):
        """Parse action string like 'move-s3-s1-a1' into (3, 1)."""
        m = re.match(r"move-s(\d+)-s(\d+)-a\d+", token)
        return (int(m.group(1)), int(m.group(2))) if m else None

    plan_transitions = [parse_action_token(a) for a in plan_actions if parse_action_token(a)]
    
    print("##############################################\nFINISHED RUNNING PLANNER\n")
    return plan_actions, plan_transitions, state_action_map

class Policy:
    """
    Policy class
    """
    def __init__(self, args, cluster_centers, transition_samples, start_state, goal_state):
        self.args = args
        
        self.pidc_x = PIDController(0.01, 0.0, 0.0)
        self.pidc_y = PIDController(0.01, 0.0, 0.4)
        
        self.cluster_centers = cluster_centers
        self.transition_samples = transition_samples
        self.start_state = int(start_state)
        self.goal_state = int(goal_state)
        
        # Initial plan from start to goal
        self.cur_plan, self.plan_transitions, _ = plan(
            self.transition_samples, self.start_state, self.goal_state, len(self.cluster_centers)
        )
        
        self.prev_state = self.start_state

    def reset(self, start_state=None, goal_state=None):
        """
        Reset internal state and optionally update start/goal.
        Call this at the beginning of each new episode.
        """
        self.pidc_x.reset()
        self.pidc_y.reset()
        
        if start_state is not None:
            self.start_state = int(start_state)
        if goal_state is not None:
            self.goal_state = int(goal_state)
        
        self.cur_plan, self.plan_transitions, _ = plan(
            self.transition_samples, self.start_state, self.goal_state, len(self.cluster_centers)
        )
        self.prev_state = self.start_state

    def _choose_pid_action(self, s_to_clust_center, obs):
        """
        PID-based action selection towards the target cluster center.
        Mirrors the logic used in eval_policy.
        """
        self.pidc_x.set_target(s_to_clust_center[0])
        self.pidc_y.set_target(s_to_clust_center[1])
        
        move_x = self.pidc_x.update(obs[0])
        move_y = 5 * self.pidc_y.update(obs[1])
        
        logits = np.array([abs(move_x), abs(move_y)])
        
        # Fallback in degenerate case
        if np.sum(logits) == 0:
            return 0
        
        act_probs = logits / np.sum(logits)
        act_idx = np.random.choice(len(act_probs), p=act_probs)
        
        if self.args.debug >= 2:
            logging.info(
                f"      currently at {obs[0:4]}, want to be at {s_to_clust_center[0:4]}, "
                f"logits = {logits}, choose on {act_idx}, probs: {act_probs}"
            )
        
        if act_idx == 0:  # x error
            return 3 if move_x > 0 else 1
        if act_idx == 1:  # y error
            return 2 if move_y > 0 else 0
        
        return 0

    def choose_action(self, obs):
        """
        Given the current observation, choose an action according to the
        current plan and PID controller, with optional correction/replanning.
        """
        if not self.plan_transitions:
            # No plan available; do nothing
            logging.error("No remaining plan transitions in Policy. Returning no-op action.")
            return 0
        
        current_state, _ = obs_to_cluster(obs, self.cluster_centers)
        current_state = int(current_state)
        
        s_from, s_to = self.plan_transitions[0]
        
        # Detect that we actually moved to a different cluster
        if current_state != self.prev_state:
            # Case 1: we arrived at the expected next state
            if current_state == s_to:
                logging.info(f"   successfully moved to {s_to}")
                self.plan_transitions.pop(0)
                
                self.pidc_x.reset()
                self.pidc_y.reset()
                
                if not self.plan_transitions:
                    # Plan is finished; do nothing
                    logging.info("Plan finished inside Policy. Returning no-op action.")
                    self.prev_state = current_state
                    return 0
                
                # Update to new first transition
                s_from, s_to = self.plan_transitions[0]
            
            # Case 2: deviated from the plan (not s_from, not s_to)
            elif current_state != s_from:
                logging.info(
                    f"   deviated from plan: expected ({s_from}->{s_to}), "
                    f"but at {current_state} instead."
                )
                
                # Decide replanning target
                if not self.args.full_replan:
                    # Replan back to the next planned state
                    replan_target = s_to
                    logging.info(f"   full_replan=False, replanning from {current_state} to {replan_target}")
                else:
                    # Replan directly to the goal
                    replan_target = self.goal_state
                    logging.info(f"   full_replan=True, replanning from {current_state} to goal {replan_target}")
                
                _, new_transitions, _ = plan(
                    self.transition_samples, current_state, replan_target, len(self.cluster_centers)
                )
                
                if not new_transitions:
                    logging.error(
                        f"   Replanning failed from {current_state} to {replan_target}. "
                        f"Returning no-op action."
                    )
                    self.plan_transitions = []
                    self.prev_state = current_state
                    return 0
                
                self.plan_transitions = new_transitions
                s_from, s_to = self.plan_transitions[0]
                
                self.pidc_x.reset()
                self.pidc_y.reset()
        
        # Now follow the plan towards s_to using PID
        s_to_clust_center = self.cluster_centers[s_to]
        action = self._choose_pid_action(s_to_clust_center, obs)
        
        if self.args.debug >= 2:
            logging.info(f"      Choosing action {action} (state {current_state}, target s{s_to})")
        
        self.prev_state = current_state
        return action
