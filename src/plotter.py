import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from collections import defaultdict


def plot_transition_graph(transition_samples):
    """
    Visualize (s, s', a) transitions as a directed weighted graph.
    Direction-specific edges are counted and curved separately if both directions exist.
    """
    # Count each unique (s, s', a)
    transition_counts = Counter(transition_samples)

    # Build directed graph with per-direction weights and actions
    G = nx.DiGraph()
    for (s, s_next, a), count in transition_counts.items():
        if G.has_edge(s, s_next):
            # multiple actions between same states → accumulate counts
            if a in G[s][s_next]['actions']:
                G[s][s_next]['actions'][a] += count
            else:
                G[s][s_next]['actions'][a] = count
            G[s][s_next]['weight'] += count
        else:
            G.add_edge(s, s_next, weight=count, actions={a: count})

    # Compute most common action per directed edge (direction-specific)
    for u, v in G.edges():
        actions = G[u][v]['actions']
        most_common_action = max(actions, key=actions.get)
        G[u][v]['action'] = most_common_action
        G[u][v]['count'] = G[u][v]['weight']

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.9)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", edgecolors="k")
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Compute curvature per direction (for bidirectional edges only)
    max_w = max(d['weight'] for _,_,d in G.edges(data=True))
    drawn_edges = set()

    for u, v in G.edges():
        # Curve if both directions exist
        if (v, u) in G.edges() and (v, u) not in drawn_edges:
            # Opposite curvatures for each direction
            for (src, dst, rad) in [(u, v, 0.2), (v, u, -0.2)]:
                weight = G[src][dst]['weight']
                action = G[src][dst]['action']
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(src, dst)],
                    width=0.5 + 2.5 * (weight / max_w),
                    alpha=0.75,
                    edge_color="gray",
                    arrows=True,
                    arrowsize=30,
                    connectionstyle=f"arc3,rad={rad}"
                )
            drawn_edges.add((u, v))
            drawn_edges.add((v, u))
        elif (v, u) not in G.edges():
            # Single direction edge → straight
            weight = G[u][v]['weight']
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=0.5 + 2.5 * (weight / max_w),
                alpha=0.75,
                edge_color="gray",
                arrows=True,
                arrowsize=30,
                connectionstyle="arc3,rad=0"
            )

    # Edge labels — direction-specific
    edge_labels = {(u, v): f"a={G[u][v]['action']}, n={G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="darkgreen")

    plt.title("Directed State Transition Graph (Separate Direction Counts)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("logs/transition_graph.png")
    #plt.show()




def plot_action_vectors(cluster_centers, transition_samples, scale=1.0):
    """
    Plot cluster centers in 2D and, for each (state, action),
    draw the average transition vector across all s --a--> s'.

    Parameters:
    - cluster_centers: array-like, shape (n_clusters, d)  (we use [:,0:2])
    - transition_samples: list of (s, s_next, a)
    - scale: multiplier to lengthen arrows visually
    """

    c = np.array(cluster_centers)
    xy = c[:, :2]   # project clusters to 2D plane

    # Collect transitions per (state, action)
    # (s, a) → list of displacement vectors: c[s'] - c[s]
    disp_dict = defaultdict(list)
    count_dict = defaultdict(int)

    for (s, s_next, a) in transition_samples:
        if s < len(xy) and s_next < len(xy):
            vec = xy[s_next] - xy[s]
            disp_dict[(s, a)].append(vec)
            count_dict[(s, a)] += 1

    # Compute averages
    avg_vec_dict = {
        key: np.mean(vecs, axis=0)
        for key, vecs in disp_dict.items()
    }

    # Plot setup
    plt.figure(figsize=(12, 9))
    plt.scatter(xy[:, 0], xy[:, 1], s=120, color='black', marker='o')
    
    # Label clusters
    for i, (x, y) in enumerate(xy):
        plt.text(x, y, f"{i}", fontsize=10, ha="center", va="center", color="white",
                 bbox=dict(facecolor="blue", edgecolor="none", pad=3))

    # Draw vectors
    for (s, a), avg_vec in avg_vec_dict.items():
        start = xy[s]
        end = start + scale * avg_vec

        # Draw arrow
        plt.arrow(
            start[0], start[1],
            scale * avg_vec[0], scale * avg_vec[1],
            head_width=0.05, length_includes_head=True,
            color="red", alpha=0.8
        )

        # Label arrow with action + count
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        plt.text(mx, my, f"a={a}\n(n={count_dict[(s,a)]})",
                 fontsize=8, color="darkred")

    plt.title("Average Action-Conditioned Transition Vectors per Cluster")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("logs/action_vector_graph.png")
    # plt.show()