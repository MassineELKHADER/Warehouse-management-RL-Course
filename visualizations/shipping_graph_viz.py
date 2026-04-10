"""
Shipping graph visualization.

Draws the transport matrix T as a directed weighted graph.
Node color: orange = hub, green = surplus, red = shortage.
Edge width and color encode shipment volume and route cost.
"""

import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_shipping_graph(
    T: np.ndarray,
    cost_matrix: np.ndarray,
    hub_fraction: float = 0.2,
    inventory: np.ndarray | None = None,
    demand: np.ndarray | None = None,
    edge_threshold: float = 0.01,
    save_path: str | None = None,
    wandb_log: bool = False,
    title: str | None = None,
) -> plt.Figure:
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    N      = T.shape[0]
    n_hubs = max(1, math.ceil(hub_fraction * N))
    T_max  = T.max() if T.max() > 0 else 1.0
    thresh = edge_threshold * T_max

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(N):
            if i != j and T[i, j] > thresh:
                G.add_edge(i, j, weight=T[i, j], cost=cost_matrix[i, j])

    if N <= 10:
        pos = nx.circular_layout(G)
    else:
        pos = nx.shell_layout(
            G, nlist=[list(range(n_hubs)), list(range(n_hubs, N))]
        )

    if inventory is not None:
        max_inv    = inventory.max() if inventory.max() > 0 else 1.0
        node_sizes = (inventory / max_inv * 1200 + 300).tolist()
    else:
        node_sizes = [600] * N

    if demand is not None and inventory is not None:
        node_colors = [
            "#d62728" if inventory[i] < demand[i] else "#2ca02c"
            for i in range(N)
        ]
    else:
        node_colors = ["#1f77b4"] * N

    for i in range(n_hubs):
        node_colors[i] = "#ff7f0e"

    edge_widths = []
    edge_colors = []
    cost_norm   = mcolors.Normalize(vmin=cost_matrix.min(), vmax=cost_matrix.max())
    cmap_edge   = cm.get_cmap("RdYlBu_r")

    for u, v, data in G.edges(data=True):
        w = data["weight"]
        edge_widths.append(1.0 + 5.0 * (w / T_max))
        edge_colors.append(cmap_edge(cost_norm(data["cost"])))

    fig, ax = plt.subplots(figsize=(max(6, N), max(5, N - 1)))

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black", linewidths=1.2,
    )

    hub_pos = {i: pos[i] for i in range(n_hubs)}
    nx.draw_networkx_nodes(
        G, hub_pos, ax=ax,
        nodelist=list(range(n_hubs)),
        node_size=[node_sizes[i] * 1.3 for i in range(n_hubs)],
        node_color=[node_colors[i] for i in range(n_hubs)],
        node_shape="*",
        edgecolors="black", linewidths=1.5,
    )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.15",
    )

    labels = {
        i: f"W{i}{'(hub)' if i < n_hubs else ''}"
        for i in range(N)
    }
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold")

    edge_labels = {
        (u, v): f"{data['weight']:.1f}"
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=7, alpha=0.8)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#ff7f0e", label="Hub"),
        Patch(facecolor="#2ca02c", label="No shortage"),
        Patch(facecolor="#d62728", label="Shortage"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=8)

    sm = cm.ScalarMappable(cmap=cmap_edge, norm=cost_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Shipping cost", shrink=0.6, pad=0.02)

    ax.set_title(title or f"Shipping policy graph (N={N})", fontsize=11)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if wandb_log:
        try:
            import wandb
            wandb.log({"viz/shipping_graph": wandb.Image(fig)})
        except Exception:
            pass

    return fig
