import os
import json
import pandas as pd

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


if __name__ == "__main__":
    results_folder_name = "results"
    df = load_all_results(results_folder_name)
    print(df.head())
    print(f"\nLoaded {len(df)} experiments.")

    # Optional: save combined CSV
    saved_csv_pth = os.path.join(results_folder_name, "all_results.csv")
    df.to_csv(saved_csv_pth, index=False)
    print(f"Saved combined results to {saved_csv_pth}")
    
    
    