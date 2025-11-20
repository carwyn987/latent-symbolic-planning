import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir="results"):
    rows = []

    # Walk recursively through all experiment subdirectories
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if not f.endswith(".json"):
                continue

            json_path = os.path.join(root, f)

            try:
                with open(json_path, "r") as fp:
                    data = json.load(fp)
            except Exception as e:
                print(f"Failed to load {json_path}: {e}")
                continue

            # Add directory metadata (optional but useful)
            data["directory"] = root
            data["filename"] = f

            rows.append(data)

    if len(rows) == 0:
        print("No result files found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df

def plot_policy_comparison(df, metric="return"):
    """
    Makes a boxplot comparing policies on a given metric:
    metric="return" or metric="steps".
    """
    sub = df[df["metric"] == metric]
    if sub.empty:
        print(f"[Warning] No rows found for metric '{metric}'")
        return

    policies = sub["policy_name"].unique()
    data = [sub[sub["policy_name"] == p]["mean"] for p in policies]

    plt.figure()
    plt.boxplot(data, labels=policies)
    plt.xlabel("Policy")
    ylabel = "Return" if metric == "return" else "Steps"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison by Policy")
    plt.tight_layout()
    plt.show()


def plot_final_policy_vs_param(df, param="num_clusters"):
    """
    Plots final_policy mean returns vs an experiment parameter
    (param must be a column in the dataframe).
    """
    returns = df[df["metric"] == "return"]
    final_df = returns[returns["policy_name"] == "final_policy"]

    if final_df.empty:
        print("[Warning] No final_policy return rows found.")
        return

    if param not in final_df.columns:
        print(f"[Warning] Parameter '{param}' not found in dataframe.")
        return

    plt.figure()
    plt.scatter(final_df[param], final_df["mean"], marker="o")
    plt.xlabel(param)
    plt.ylabel("Mean Return")
    plt.title(f"Final Policy Return vs {param}")
    plt.tight_layout()
    plt.show()


def run_analysis(df):
    """Runs all analysis plots."""
    print("\nRunning analysis...")

    # Compare policies on returns and on steps
    plot_policy_comparison(df, metric="return")
    plot_policy_comparison(df, metric="steps")

    # Sensitivity plots (add more parameters if desired)
    plot_final_policy_vs_param(df, param="num_clusters")
    plot_final_policy_vs_param(df, param="num_steps")
    plot_final_policy_vs_param(df, param="num_act_apply")
    plot_final_policy_vs_param(df, param="clustering_method")

    print("Analysis complete.")

if __name__ == "__main__":
    results_folder_name = "results"
    df = load_all_results(results_folder_name)
    print(df.head())
    print(f"\nLoaded {len(df)} experiments.")

    # Optional: save combined CSV
    saved_csv_pth = os.path.join(results_folder_name, "all_results.csv")
    df.to_csv(saved_csv_pth, index=False)
    print(f"Saved combined results to {saved_csv_pth}")
    
    run_analysis(df)