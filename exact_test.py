import pickle
import time
from itertools import combinations
from pathlib import Path

import networkx as nx


DATASET_DIR = Path(__file__).parent / "Dataset" / "quantum_dataset"
INPUT_FILES = (
    "quantum_n10_d0.2.pkl",
    "quantum_n10_d0.5.pkl",
    "quantum_n10_d0.8.pkl",
)


def cut_value(graph: nx.Graph, placed: set[int]) -> int:
    """Return the number of edges crossing the placed/unplaced cut."""
    return sum((u in placed) != (v in placed) for u, v in graph.edges)


def solve_graph(graph_data: dict) -> tuple[int, list[int]]:
    """Find an exact minimum linear arrangement value and one optimal ordering."""
    num_vertices = graph_data["num_vertices"]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    graph.add_edges_from(graph_data["edges"])

    # dp[prefix] is the minimum sum of cuts after placing prefix.
    dp: dict[tuple[int, ...], int] = {(): 0}
    parent: dict[tuple[int, ...], int] = {}
    vertices = tuple(range(num_vertices))

    for size in range(1, num_vertices + 1):
        for prefix in combinations(vertices, size):
            prefix_set = set(prefix)
            best_value = float("inf")
            best_vertex = None

            for vertex in prefix:
                previous = tuple(v for v in prefix if v != vertex)
                value = dp[previous] + cut_value(graph, prefix_set)
                if value < best_value:
                    best_value = value
                    best_vertex = vertex

            dp[prefix] = int(best_value)
            parent[prefix] = best_vertex

    full_prefix = vertices
    ordering: list[int] = []
    prefix = full_prefix
    while prefix:
        vertex = parent[prefix]
        ordering.append(vertex)
        prefix = tuple(v for v in prefix if v != vertex)
    ordering.reverse()

    return dp[full_prefix], ordering


def process_dataset(input_path: Path) -> Path:
    with input_path.open("rb") as file:
        dataset = pickle.load(file)

    start = time.perf_counter()
    results = []
    for graph_data in dataset["graphs"]:
        optimal_value, optimal_ordering = solve_graph(graph_data)
        results.append(
            {
                "id": graph_data["id"],
                "optimal_value": optimal_value,
                "optimal_ordering": optimal_ordering,
            }
        )

    output = {
        "source_file": input_path.name,
        "num_vertices": dataset["num_vertices"],
        "density": dataset["density"],
        "num_graphs": dataset["num_graphs"],
        "seed": dataset["seed"],
        "graphs": results,
    }
    output_path = input_path.with_name(f"{input_path.stem}_optimal.pkl")
    with output_path.open("wb") as file:
        pickle.dump(output, file)

    elapsed = time.perf_counter() - start
    print(f"{input_path.name}: {len(results)} graphs solved in {elapsed:.4f} seconds")
    print(f"  optimal values: {[result['optimal_value'] for result in results]}")
    print(f"  saved to: {output_path}")
    return output_path


def main() -> None:
    for filename in INPUT_FILES:
        process_dataset(DATASET_DIR / filename)


if __name__ == "__main__":
    main()
