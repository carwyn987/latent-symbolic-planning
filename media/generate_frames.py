import ast
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# 1. Parse trajectory log
# ============================================================

def load_traj_log(path):
    with open(path, "r") as f:
        lines = [l.rstrip() for l in f]

    # cluster centers
    raw = next(l for l in lines if l.startswith("cluster_centers=")).split("=",1)[1]
    cc = np.array([coords for (_cid, coords) in ast.literal_eval(raw)])

    # steps
    steps = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("STEP"):
            pos_vals = lines[i+1].split("=",1)[1]
            pos = np.array([float(x) for x in pos_vals.strip("[]").split()])

            plan = ast.literal_eval(lines[i+3].split("=",1)[1])
            steps.append({
                "pos": pos,
                "plan": plan
            })
            i += 4
        else:
            i += 1

    return cc, steps


# ============================================================
# 2. Simple draw with scale + center offset
# ============================================================

def draw_frame(ax, cc, step, scale, center_x, center_y, crop):
    ax.clear()
    ax.axis("off")

    # world → pixel transform
    def T(pt):
        return np.array([
            pt[0] * scale + center_x,
            pt[1] * scale + center_y
        ])

    # cluster centers
    pts = np.array([T(p) for p in cc])
    
    ax.scatter(pts[:,0], pts[:,1], s=80, color="black")

    for i, (x, y) in enumerate(pts):
        ax.text(x, y, str(i), ha="center", va="center",
                color="white",
                bbox=dict(facecolor="blue", pad=2))

    # plan arrows
    for s, s_next in step["plan"]:
        if s < len(cc) and s_next < len(cc):
            a = T(cc[s])
            b = T(cc[s_next])
            v = b - a
            ax.arrow(a[0], a[1], v[0], v[1],
                     color="red",
                     width=0.2,
                     head_width=0.5,
                     length_includes_head=True)

    # agent
    px = T(step["pos"])
    # ax.scatter(px[0], px[1], s=50, color="cyan", edgecolor="black")


# ============================================================
# 3. Main frame generator
# ============================================================

def create_overlay_frames(
        log_path,
        out_dir="media/frames",
        fig_size=(8,6),
        scale=20,
        crop=(0,0)      # left, right, bottom, top pixels
    ):

    os.makedirs(out_dir, exist_ok=True)
    cc, steps = load_traj_log(log_path)
    
    # precompute center in figure pixel coordinates
    dpi = 90
    fig_width_px  = fig_size[0] * dpi
    fig_height_px = fig_size[1] * dpi

    center_x = fig_width_px  / 2
    center_y = fig_height_px / 2

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_xlim(0, fig_width_px)
    ax.set_ylim(0, fig_height_px)

    # invert y-axis because image coords have (0,0) at top-left
    ax.invert_yaxis()

    print("Saving frames →", out_dir)

    for i, step in enumerate(tqdm(steps)):
        draw_frame(ax, cc, step, scale, center_x, center_y, crop)

        # save full fig
        fname = f"{out_dir}/frame_{i:06d}.png"
        fig.savefig(fname, transparent=True)

    print(f"Saved {len(steps)} frames.")


# ============================================================
# Example run
# ============================================================

if __name__ == "__main__":
    create_overlay_frames(
        "logs/traj_log_26.txt",
        fig_size=(8, 4),   # inches
        scale=20,          # world→pixel
        crop=(0.6,0.1)
    )
