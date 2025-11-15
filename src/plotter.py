import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

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
    #plt.show()
