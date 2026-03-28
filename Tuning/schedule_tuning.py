import os
import pickle
import time
import sys
import numpy as np
import pandas as pd
import networkx as nx
from dwave.samplers import PathIntegralAnnealingSampler

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Utils.MinLA as minla

# Construct dataset path relative to parent directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset", "quantum_dataset")
RESULTS_DIR = os.path.join(PARENT_DIR, "Results")


def read_dataset():
    """Read all pickle files from quantum_dataset."""
    datasets = {}
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".pkl"):
            filepath = os.path.join(DATASET_PATH, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                datasets[data['num_vertices']] = data
    return datasets


def convert_graph_data_to_nx(graph_data):
    """Convert graph data to NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(graph_data['num_vertices']))
    G.add_edges_from(graph_data['edges'])
    return G


def create_schedules(num_sweeps):
    """Create all beta schedule types."""
    schedules = {}
    
    # Linear schedules
    schedules['Linear'] = (np.linspace(0, 1, num=num_sweeps), np.linspace(1, 0, num=num_sweeps))
    
    # Geometric schedules
    schedules['Geometric'] = (np.geomspace(1e-6, 1, num=num_sweeps), np.geomspace(1, 1e-6, num=num_sweeps))
    
    # Power schedules
    schedules['Power (1/3)'] = (np.power(np.linspace(0, 1, num_sweeps), 1/3), np.power(np.linspace(1, 0, num_sweeps), 1/3))
    schedules['Power (1/2)'] = (np.power(np.linspace(0, 1, num_sweeps), 1/2), np.power(np.linspace(1, 0, num_sweeps), 1/2))
    schedules['Power (2)'] = (np.power(np.linspace(0, 1, num_sweeps), 2), np.power(np.linspace(1, 0, num_sweeps), 2))
    schedules['Power (3)'] = (np.power(np.linspace(0, 1, num_sweeps), 3), np.power(np.linspace(1, 0, num_sweeps), 3))

    # Trigonometric schedules
    schedules['Trigonometric'] = (np.sin(np.pi / 2 * np.linspace(0, 1, num_sweeps))**2, np.sin(np.pi / 2 * np.linspace(1, 0, num_sweeps))**2)
    
    # Sigmoid schedules
    sigmoid = _sigmoid_schedule(num_sweeps)
    schedules['Sigmoid'] = (sigmoid, 1 - sigmoid)
    
    # Logarithmic schedules
    logarithmic = _logarithmic_schedule(num_sweeps)
    schedules['Logarithmic'] = (logarithmic, 1 - logarithmic)
    
    # Fixed Hd with different Hp schedules
    hd = np.ones(num_sweeps)
    schedules['Fixed Hd, Linear Hp'] = (schedules['Linear'][0], hd)
    schedules['Fixed Hd, Geometric Hp'] = (schedules['Geometric'][0], hd)
    schedules['Fixed Hd, Power Hp (1/2)'] = (schedules['Power (1/2)'][0], hd)
    schedules['Fixed Hd, Power Hp (2)'] = (schedules['Power (2)'][0], hd)
    schedules['Fixed Hd, Trigonometric Hp'] = (schedules['Trigonometric'][0], hd)
    schedules['Fixed Hd, Sigmoid Hp'] = (schedules['Sigmoid'][0], hd)
    schedules['Fixed Hd, Logarithmic Hp'] = (schedules['Logarithmic'][0], hd)
    
    return schedules


def _sigmoid_schedule(num_steps, k=10):
    """Generate sigmoid schedule."""
    x = np.linspace(0, 1, num_steps)
    s = 1 / (1 + np.exp(-k * (x - 0.5)))
    return (s - s.min()) / (s.max() - s.min())


def _logarithmic_schedule(num_steps):
    """Generate logarithmic schedule."""
    t = np.arange(1, num_steps + 1)
    s = np.log(t + 1)
    return (s - s.min()) / (s.max() - s.min())


def decode_solution(raw_sample, n):
    """Decode solution from raw sample."""
    sol = np.array([
        [raw_sample.get(f'X[{u}][{k}]', 0) for k in range(n)]
        for u in range(n)
    ], dtype=int)
    is_feasible = check_feasibility(sol, n)
    ordering = np.sum(sol, axis=1)
    return ordering, is_feasible


def check_feasibility(sol, n):
    """Check if solution is feasible."""
    for u in range(n):
        if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
            return False
    labels = set(np.sum(sol, axis=1))
    return labels == set(range(1, n + 1))


def run_schedule_tuning_experiment():
    """Run schedule tuning experiment on a 30-vertex graph from quantum_dataset with all schedules."""
    SEEDS = [42, 123, 456, 789, 999]
    num_sweeps = 1000
    num_reads = 10
    
    # Load dataset
    print("Loading quantum_dataset...")
    datasets = read_dataset()
    
    n = 25
    
    # Check if 25-vertex dataset exists
    if n not in datasets:
        print(f"Error: No {n}-vertex graphs in dataset. Available sizes: {sorted(datasets.keys())}")
        return
    
    # Get first 25-vertex graph
    print(f"Loading {n}-vertex graph from quantum_dataset...")
    graph_data = datasets[25]['graphs'][0]
    G = convert_graph_data_to_nx(graph_data)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    print(f"Graph: {n} vertices, {m} edges")
    
    # Create BQM
    bqm = minla.generate_bqm_instance(G)
    
    # Create all schedules
    print(f"Creating {len(create_schedules(num_sweeps))} schedules...")
    schedules = create_schedules(num_sweeps)
    
    # Define which schedules are standard (forward/inverse pairs) vs fixed Hd
    schedule_names = [
        'Linear', 'Geometric', 'Power (1/3)', 'Power (1/2)', 'Power (2)', 'Power (3)',
        'Trigonometric', 'Sigmoid', 'Logarithmic', 'Fixed Hd, Linear Hp', 'Fixed Hd, Geometric Hp', 'Fixed Hd, Power Hp (1/2)',
        'Fixed Hd, Power Hp (2)', 'Fixed Hd, Trigonometric Hp', 'Fixed Hd, Sigmoid Hp',
        'Fixed Hd, Logarithmic Hp'
    ]
    
    all_rows = []
    
    print(f"\nTesting {len(schedules)} schedules on {n}-vertex graph with 5 seeds...\n")
    
    # Test each schedule (forward and inverse for standard, just forward for fixed Hd)
    schedule_counter = 0
    
    # Test standard schedules (both forward and inverse)
    for sched_base_name in schedule_names:
        sched_name = sched_base_name
        if sched_name not in schedules:
            continue
        
        schedule_counter += 1
        print(f"[{schedule_counter}/{len(schedules)}] Testing {sched_name}...")
        
        seed_results = []
        
        for seed in SEEDS:
            np.random.seed(seed)
            
            t0 = time.time()
            
            solver = PathIntegralAnnealingSampler()
            
            hp_field, hd_field = schedules[sched_name]
            
            sampleset = solver.sample(
                bqm,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                beta_schedule_type='custom',
                Hp_field=hp_field,
                Hd_field=hd_field,
                seed=seed
            )
            
            elapsed = time.time() - t0
            
            best = sampleset.first
            energy = best.energy
            ordering, feasible = decode_solution(best.sample, n)
            
            seed_results.append({
                'seed': seed,
                'energy': energy,
                'feasible': feasible,
                'time_s': elapsed,
                'ordering': ordering
            })
        
        # Select best result across seeds
        best_result = None
        for result in seed_results:
            if best_result is None:
                best_result = result
            elif result['feasible'] and not best_result['feasible']:
                best_result = result
            elif result['feasible'] and best_result['feasible']:
                # Both feasible - need to evaluate ordering quality
                pass
            elif not result['feasible'] and not best_result['feasible']:
                if result['energy'] < best_result['energy']:
                    best_result = result
        
        row = {
            'schedule': sched_name,
            'n_vertices': n,
            'm_edges': m,
            'energy': best_result['energy'],
            'feasible': best_result['feasible'],
            'time_s': round(best_result['time_s'], 3),
            'best_seed': best_result['seed'],
        }
        all_rows.append(row)
        print(f"  ✓ Best seed: {best_result['seed']}, Energy: {best_result['energy']:.4f}, Feasible: {best_result['feasible']}, Time: {best_result['time_s']:.2f}s")

    
    # Save results
    df = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"schedule_tuning_{n}v_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {csv_path}")
    print(f"{'='*70}")
    
    # Print summary
    print("\nSchedule Tuning Results Summary:")
    print(df.to_string(index=False))
    
    # Find best schedule
    best_idx = df['energy'].idxmin()
    print(f"\nBest schedule: {df.loc[best_idx, 'schedule']}")
    print(f"Best energy: {df.loc[best_idx, 'energy']:.4f}")
    print(f"Feasible: {df.loc[best_idx, 'feasible']}")


if __name__ == "__main__":
    run_schedule_tuning_experiment()
