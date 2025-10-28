import subprocess
import json, os
from src.pddl_convert import write_ppddl_files
import re

def plan(transitions, start_state, goal_state, num_states):
    
    # Create plan files
    output_dir = "ppddl_output"
    problem_out = "p01"
    domain_out = "d01"
    
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

    # print(f"Plan contains {len(plan_actions)} actions:")
    # for a in plan_actions:
    #     print(" ", a)
    
    # --------------------------------------------------------------------------
    # 3. Build state→action mapping from learned transitions
    # --------------------------------------------------------------------------
    state_action_map = {(s_from, s_to): act for s_from, s_to, act in transitions}

    def parse_action_token(token):
        """Parse action string like 'move-s3-s1-a1' into (3, 1)."""
        m = re.match(r"move-s(\d+)-s(\d+)-a\d+", token)
        return (int(m.group(1)), int(m.group(2))) if m else None

    plan_transitions = [parse_action_token(a) for a in plan_actions if parse_action_token(a)]
    
    return plan_actions, plan_transitions, state_action_map