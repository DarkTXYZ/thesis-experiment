import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Utils.MinLA import generate_bqm_instance

DATASET_PATH = os.path.join(ROOT_DIR, "Dataset/quantum_dataset/quantum_extra.pkl")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Results/plots")

NUM_VERTICES = 10
GRAPH_INDEX = 0
DENSITY = 0.5
RANDOM_SEED = 42

_VAR_RE = re.compile(r"X\[(\d+)\]\[(\d+)\]")


def generate_connected_random_graph(n: int, density: float, seed: int, max_attempts: int = 1000) -> nx.Graph:
    for attempt in range(max_attempts):
        G = nx.gnp_random_graph(n, density, seed=seed + attempt)
        if nx.is_connected(G):
            return G
    raise RuntimeError(f"Could not generate a connected graph with n={n}, density={density}")


def load_graph(dataset_path: str, num_vertices: int, graph_index: int) -> nx.Graph:
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    if num_vertices not in data:
        print(f"No dataset entry for num_vertices={num_vertices}; generating a random connected graph instead.")
        G = generate_connected_random_graph(num_vertices, DENSITY, RANDOM_SEED)
        G.graph["id"] = "random"
        return G

    graph_data = data[num_vertices]["graphs"][graph_index]

    G = nx.Graph()
    G.add_nodes_from(range(graph_data["num_vertices"]))
    G.add_edges_from(graph_data["edges"])
    G.graph["id"] = graph_data["id"]
    return G


def build_q_matrix(bqm) -> tuple[np.ndarray, list[str]]:
    """Assemble a dense, upper-triangular Q matrix (standard QUBO convention:
    diagonal = linear biases, upper off-diagonal = quadratic biases), with
    variables ordered by (node, thermometer bit) so node blocks are visible.
    """
    variables = sorted(bqm.variables, key=lambda v: tuple(map(int, _VAR_RE.match(v).groups())))
    index = {var: i for i, var in enumerate(variables)}

    n = len(variables)
    Q = np.full((n, n), np.nan)

    for var, bias in bqm.linear.items():
        i = index[var]
        Q[i, i] = bias

    for (u, v), bias in bqm.quadratic.items():
        i, j = index[u], index[v]
        Q[min(i, j), max(i, j)] = bias

    return Q, variables


def plot_q_matrix(Q: np.ndarray, num_vertices: int, graph_id: int, num_edges: int, out_path: str) -> None:
    n = Q.shape[0]
    vmax = np.nanmax(np.abs(Q))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f0efec")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(Q, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))

    # Delineate each node's block of thermometer variables.
    block = n // num_vertices
    for boundary in range(block, n, block):
        ax.axhline(boundary - 0.5, color="#c3c2b7", linewidth=0.6)
        ax.axvline(boundary - 0.5, color="#c3c2b7", linewidth=0.6)

    tick_positions = [block * u + block / 2 - 0.5 for u in range(num_vertices)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(range(num_vertices), fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(range(num_vertices), fontsize=7)
    ax.set_xlabel("node u")
    ax.set_ylabel("node u")

    ax.set_title(
        f"QUBO Q-matrix — MinLA instance (n={num_vertices}, "
        f"graph id={graph_id}, |E|={num_edges}, {n} variables)"
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Q coefficient")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_q_graph(bqm) -> tuple[nx.Graph, list[str]]:
    """Represent the QUBO as a node-link graph: one node per binary variable
    (grid-positioned by (node u, thermometer bit k)), one edge per nonzero
    quadratic coupling.
    """
    variables = sorted(bqm.variables, key=lambda v: tuple(map(int, _VAR_RE.match(v).groups())))
    index = {var: i for i, var in enumerate(variables)}

    Gq = nx.Graph()
    for i, var in enumerate(variables):
        u, k = map(int, _VAR_RE.match(var).groups())
        Gq.add_node(i, u=u, k=k, bias=bqm.linear.get(var, 0.0))

    for (a, b), bias in bqm.quadratic.items():
        if bias != 0:
            Gq.add_edge(index[a], index[b], bias=bias)

    return Gq, variables


def plot_q_graph(Gq: nx.Graph, num_vertices: int, graph_id: int, num_edges: int, out_path: str) -> None:
    pos = {i: (data["k"], -data["u"]) for i, data in Gq.nodes(data=True)}

    positive = "#e34948"
    negative = "#2a78d6"

    node_colors = [positive if Gq.nodes[i]["bias"] > 0 else negative for i in Gq.nodes]
    edge_colors = [positive if Gq.edges[e]["bias"] > 0 else negative for e in Gq.edges]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    nx.draw_networkx_edges(Gq, pos, ax=ax, edge_color=edge_colors, width=0.4, alpha=0.35)
    nx.draw_networkx_nodes(Gq, pos, ax=ax, node_color=node_colors, node_size=25, edgecolors="none")

    block = Gq.number_of_nodes() // num_vertices
    ax.set_xticks(range(block))
    ax.set_xticklabels(range(block), fontsize=7)
    ax.set_yticks([-u for u in range(num_vertices)])
    ax.set_yticklabels(range(num_vertices), fontsize=7)
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.set_ylabel("node u")
    ax.set_xlabel("thermometer bit k")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=positive, markersize=7, label="positive"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=negative, markersize=7, label="negative"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_title(
        f"QUBO coupling graph — MinLA instance (n={num_vertices}, graph id={graph_id}, |E|={num_edges})\n"
        f"{Gq.number_of_nodes()} variables, {Gq.number_of_edges()} couplings",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    G = load_graph(DATASET_PATH, NUM_VERTICES, GRAPH_INDEX)
    bqm = generate_bqm_instance(G)
    bqm.normalize()
    Q, variables = build_q_matrix(bqm)
    Gq, _ = build_q_graph(bqm)

    print(f"Graph: n={G.number_of_nodes()}, |E|={G.number_of_edges()}, id={G.graph['id']}")
    print(f"BQM: {len(variables)} variables, {bqm.num_interactions} quadratic terms, offset={bqm.offset}")

    matrix_path = os.path.join(OUTPUT_DIR, f"q_matrix_n{NUM_VERTICES}_g{G.graph['id']}.png")
    plot_q_matrix(Q, NUM_VERTICES, G.graph["id"], G.number_of_edges(), matrix_path)
    print(f"Saved matrix plot to {matrix_path}")

    graph_path = os.path.join(OUTPUT_DIR, f"q_graph_n{NUM_VERTICES}_g{G.graph['id']}.png")
    plot_q_graph(Gq, NUM_VERTICES, G.graph["id"], G.number_of_edges(), graph_path)
    print(f"Saved graph plot to {graph_path}")


if __name__ == "__main__":
    main()
