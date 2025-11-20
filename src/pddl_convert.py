import os
from typing import List, Tuple

def write_ppddl_files(
    transitions: List[Tuple[int, int, int]],
    num_states: int,
    start_state: int,
    goal_state: int,
    output_dir: str = "ppddl_output",
    domain_name: str = "d01",
    problem_name: str = "p01"
):
    """
    Convert learned transition information into PPDDL domain and problem files.

    Args:
        transitions: list of (state_from, state_to, action_id) tuples
        num_states: total number of clustered states
        start_state: integer ID of start cluster
        goal_state: integer ID of goal cluster
        output_dir: where to save .pddl files
        domain_name: name of domain (used inside the PPDDL definition)
        problem_name: name of problem instance
    """
    os.makedirs(output_dir, exist_ok=True)

    domain_file = os.path.join(output_dir, f"{domain_name}.pddl")
    problem_file = os.path.join(output_dir, f"{problem_name}.pddl")

    # ------------- DOMAIN FILE -------------
    with open(domain_file, "w") as f:
        f.write(f"(define (domain {domain_name})\n")
        f.write("  (:requirements :strips :typing :negative-preconditions :equality)\n")
        f.write("  (:types state)\n")
        f.write("  (:predicates (at ?s - state))\n\n")

        for (s_from, s_to, a_id) in transitions:
            f.write(f"  (:action move-s{s_from}-s{s_to}-a{a_id}\n")
            f.write("    :parameters ()\n")
            f.write(f"    :precondition (at s{s_from})\n")
            f.write(f"    :effect (and (not (at s{s_from})) (at s{s_to})))\n\n")

        f.write(")\n")

    # ------------- PROBLEM FILE -------------
    with open(problem_file, "w") as f:
        f.write(f"(define (problem {problem_name})\n")
        f.write(f"  (:domain {domain_name})\n")
        f.write("  (:objects\n")
        f.write("    " + " ".join([f"s{i}" for i in range(num_states)]) + " - state\n")
        f.write("  )\n")
        f.write("  (:init\n")
        f.write(f"    (at s{start_state})\n")
        f.write("  )\n")
        f.write("  (:goal\n")
        f.write(f"    (at s{goal_state})\n")
        f.write("  )\n")
        f.write(")\n")

    print(f"✅ PPDDL files written to: {output_dir}")
    print(f" - Domain: {domain_file}")
    print(f" - Problem: {problem_file}")
