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
    
    # Exponential schedules (fast cooling)
    exponential = _exponential_schedule(num_sweeps)
    schedules['Exponential'] = (exponential, 1 - exponential)
    
    # Cosine annealing schedules
    cosine = _cosine_schedule(num_sweeps)
    schedules['Cosine'] = (cosine, 1 - cosine)
    
    # Square root schedules
    schedules['Power (1/4)'] = (np.power(np.linspace(0, 1, num_sweeps), 1/4), np.power(np.linspace(1, 0, num_sweeps), 1/4))
    schedules['Power (4)'] = (np.power(np.linspace(0, 1, num_sweeps), 4), np.power(np.linspace(1, 0, num_sweeps), 4))
    schedules['Power (5)'] = (np.power(np.linspace(0, 1, num_sweeps), 5), np.power(np.linspace(1, 0, num_sweeps), 5))
    
    # Hyperbolic schedules
    hyperbolic = _hyperbolic_schedule(num_sweeps)
    schedules['Hyperbolic'] = (hyperbolic, 1 - hyperbolic)
    
    # Piecewise schedules (steep start, gradual end)
    piecewise = _piecewise_schedule(num_sweeps)
    schedules['Piecewise (Fast-Slow)'] = (piecewise, 1 - piecewise)
    
    # Fixed Hd with different Hp schedules
    hd = np.ones(num_sweeps)
    schedules['Fixed Hd, Linear Hp'] = (schedules['Linear'][0], hd)
    schedules['Fixed Hd, Geometric Hp'] = (schedules['Geometric'][0], hd)
    schedules['Fixed Hd, Power Hp (1/2)'] = (schedules['Power (1/2)'][0], hd)
    schedules['Fixed Hd, Power Hp (2)'] = (schedules['Power (2)'][0], hd)
    schedules['Fixed Hd, Trigonometric Hp'] = (schedules['Trigonometric'][0], hd)
    schedules['Fixed Hd, Sigmoid Hp'] = (schedules['Sigmoid'][0], hd)
    schedules['Fixed Hd, Logarithmic Hp'] = (schedules['Logarithmic'][0], hd)
    schedules['Fixed Hd, Exponential Hp'] = (schedules['Exponential'][0], hd)
    schedules['Fixed Hd, Cosine Hp'] = (schedules['Cosine'][0], hd)

    # Fixed Hp with different Hp schedules
    hp = np.ones(num_sweeps)
    schedules['Fixed Hp, Linear Hd'] = (hp, schedules['Linear'][1])
    schedules['Fixed Hp, Geometric Hd'] = (hp, schedules['Geometric'][1])
    schedules['Fixed Hp, Power Hd (1/2)'] = (hp, schedules['Power (1/2)'][1])
    schedules['Fixed Hp, Power Hd (2)'] = (hp, schedules['Power (2)'][1])
    schedules['Fixed Hp, Trigonometric Hd'] = (hp, schedules['Trigonometric'][1])
    schedules['Fixed Hp, Sigmoid Hd'] = (hp, schedules['Sigmoid'][1])
    schedules['Fixed Hp, Logarithmic Hd'] = (hp, schedules['Logarithmic'][1])
    schedules['Fixed Hp, Exponential Hd'] = (hp, schedules['Exponential'][1])
    schedules['Fixed Hp, Cosine Hd'] = (hp, schedules['Cosine'][1])

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


def _exponential_schedule(num_steps, base=2.0):
    """Generate exponential schedule (fast cooling)."""
    t = np.linspace(0, 1, num_steps)
    s = (np.exp(base * t) - 1) / (np.exp(base) - 1)
    return (s - s.min()) / (s.max() - s.min())


def _cosine_schedule(num_steps):
    """Generate cosine annealing schedule."""
    t = np.linspace(0, 1, num_steps)
    s = 0.5 * (1 + np.cos(np.pi * t))
    return (s - s.min()) / (s.max() - s.min())


def _hyperbolic_schedule(num_steps):
    """Generate hyperbolic schedule (smooth transitions)."""
    t = np.linspace(0, 2, num_steps)
    s = np.tanh(t - 1)
    return (s - s.min()) / (s.max() - s.min())


def _piecewise_schedule(num_steps):
    """Generate piecewise schedule (fast start, gradual end)."""
    mid = num_steps // 2
    s = np.concatenate([
        np.linspace(0, 0.8, mid),  # Fast initial cooling to 80%
        np.linspace(0.8, 1, num_steps - mid)  # Slow final refinement
    ])
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
    """Run schedule tuning experiment on five graphs from quantum_dataset with all schedules."""
    SEEDS = [42, 123, 456, 789, 999]
    num_sweeps = 1000
    num_reads = 10
    NUM_GRAPHS = 5  # Test on 1 different graph

    # Load dataset
    print("Loading quantum_dataset...")
    datasets = read_dataset()
    
    n = 25
    normalized = True
    
    # Check if 25-vertex dataset exists
    if n not in datasets:
        print(f"Error: No {n}-vertex graphs in dataset. Available sizes: {sorted(datasets.keys())}")
        return
    
    # Get first NUM_GRAPHS 25-vertex graphs
    graphs_data = datasets[25]['graphs'][:NUM_GRAPHS]
    if len(graphs_data) < NUM_GRAPHS:
        print(f"Warning: Only {len(graphs_data)} graphs available, using all of them")
        NUM_GRAPHS = len(graphs_data)
    
    print(f"Loading {NUM_GRAPHS} {n}-vertex graphs from quantum_dataset...")
    
    # Create all schedules
    print(f"Creating {len(create_schedules(num_sweeps))} schedules...")
    schedules = create_schedules(num_sweeps)
    
    # Define schedule names
    schedule_names = [
        'Linear', 'Geometric', 'Power (1/3)', 'Power (1/4)', 'Power (1/2)', 'Power (2)', 'Power (3)', 'Power (4)', 'Power (5)',
        'Trigonometric', 'Sigmoid', 'Logarithmic', 'Exponential', 'Cosine', 'Hyperbolic', 'Piecewise (Fast-Slow)',
        'Fixed Hd, Linear Hp', 'Fixed Hd, Geometric Hp', 'Fixed Hd, Power Hp (1/2)',
        'Fixed Hd, Power Hp (2)', 'Fixed Hd, Trigonometric Hp', 'Fixed Hd, Sigmoid Hp',
        'Fixed Hd, Logarithmic Hp', 'Fixed Hd, Exponential Hp', 'Fixed Hd, Cosine Hp',
        'Fixed Hp, Linear Hd', 'Fixed Hp, Geometric Hd', 'Fixed Hp, Power Hd (1/2)',
        'Fixed Hp, Power Hd (2)', 'Fixed Hp, Trigonometric Hd', 'Fixed Hp, Sigmoid Hd',
        'Fixed Hp, Logarithmic Hd', 'Fixed Hp, Exponential Hd', 'Fixed Hp, Cosine Hd'
    ]
    
    # Check for existing results
    existing_schedules = set()
    all_rows = []
    
    # Filter results files based on normalization status
    all_results_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith(f"schedule_tuning_{NUM_GRAPHS}graphs_{n}v_")], reverse=True)
    
    # Only use results files that match the normalization status
    if normalized:
        results_files = [f for f in all_results_files if 'normalized' in f]
        print(f"Looking for normalized results files...")
    else:
        results_files = [f for f in all_results_files if 'normalized' not in f]
        print(f"Looking for non-normalized results files...")
    
    if results_files:
        latest_results = os.path.join(RESULTS_DIR, results_files[0])
        normalization_status = "normalized" if normalized else "non-normalized"
        print(f"Found existing {normalization_status} results: {results_files[0]}")
        existing_df = pd.read_csv(latest_results)
        existing_schedules = set(existing_df['schedule'].unique())
        all_rows = existing_df.to_dict('records')
        print(f"Loaded {len(existing_schedules)} previously tested schedules from {normalization_status} results")
    else:
        if all_results_files:
            print(f"⚠️  Warning: No {('normalized' if normalized else 'non-normalized')} results found.")
            print(f"    Available results: {[f for f in all_results_files]}")
            print(f"    Starting fresh with {('normalized' if normalized else 'non-normalized')} BQM experiments.")
        else:
            print(f"No existing results found. Starting fresh {('normalized' if normalized else 'non-normalized')} BQM experiments.")
    
    # Filter to only untested schedules
    new_schedules = [s for s in schedule_names if s not in existing_schedules]
    
    print(f"\nSchedules to test: {len(new_schedules)} new, {len(existing_schedules)} already tested")
    
    if not new_schedules:
        print("✓ All schedules have been tested!")
        df = pd.DataFrame(all_rows)
        print("\nSchedule Tuning Results Summary:")
        print(df.to_string(index=False))
        df_sorted = df.sort_values('feasible_percentage', ascending=False)
        print(f"\n{'='*70}")
        print("Top 5 schedules by feasibility:")
        print(df_sorted[['schedule', 'feasible_instances', 'total_instances', 'feasible_percentage']].head(10).to_string(index=False))
        print(f"{'='*70}")
        return
    
    print(f"\nTesting {len(new_schedules)} new schedules on {NUM_GRAPHS} graphs with {len(SEEDS)} seeds each...\n")
    
    # Test each schedule
    schedule_counter = 0
    
    for sched_name in new_schedules:
        if sched_name not in schedules:
            continue
        
        schedule_counter += 1
        print(f"[{schedule_counter}/{len(new_schedules)}] Testing {sched_name}...")
        
        feasible_energies_per_instance = []  # Store best feasible energy for each instance (graph)
        total_runs = 0      # Total number of runs
        total_time = 0      # Total computation time
        feasible_count = 0  # Count of instances with feasible solutions
        
        # Test on each of the NUM_GRAPHS graphs
        for graph_idx, graph_data in enumerate(graphs_data):
            G = convert_graph_data_to_nx(graph_data)
            num_vertices = G.number_of_nodes()
            num_edges = G.number_of_edges()
            
            # Create BQM
            bqm = minla.generate_bqm_instance(G)
            if normalized:
                bqm.normalize()
            
            feasible_energies_per_seed = []  # Store best feasible energy for each seed
            
            # Test with each seed
            for seed in SEEDS:
                np.random.seed(seed)
                total_runs += 1
                
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
                total_time += elapsed
                
                # Find the best feasible solution in the sampleset
                best_feasible_energy = None
                for sample, energy in zip(sampleset.samples(), sampleset.data_vectors['energy']):
                    ordering, feasible = decode_solution(sample, num_vertices)
                    if feasible:
                        if best_feasible_energy is None or energy < best_feasible_energy:
                            best_feasible_energy = energy
                
                if best_feasible_energy is not None:
                    feasible_energies_per_seed.append(best_feasible_energy)
            
            # Average the best feasible energies across seeds for this instance
            if feasible_energies_per_seed:
                avg_feasible_energy = np.mean(feasible_energies_per_seed)
                feasible_energies_per_instance.append(avg_feasible_energy)
                feasible_count += 1
        
        # Calculate overall statistics
        avg_time = total_time / total_runs
        feasible_percentage = (feasible_count / NUM_GRAPHS) * 100
        
        # Create row with individual energies for each graph
        row = {
            'schedule': sched_name,
            'n_graphs': NUM_GRAPHS,
            'n_vertices': n,
            'feasible_instances': feasible_count,
            'total_instances': NUM_GRAPHS,
            'feasible_percentage': round(feasible_percentage, 2),
            'avg_time_s': round(avg_time, 3),
        }
        
        # Add individual energies for each graph
        for i in range(NUM_GRAPHS):
            if i < len(feasible_energies_per_instance):
                row[f'graph_{i+1}_energy'] = round(feasible_energies_per_instance[i], 4)
            else:
                row[f'graph_{i+1}_energy'] = None
        
        all_rows.append(row)
        energy_str = ', '.join([f"{e:.4f}" if e is not None else "N/A" for e in feasible_energies_per_instance])
        print(f"  ✓ Feasible instances: {feasible_count}/{NUM_GRAPHS} ({feasible_percentage:.1f}%), Energies: [{energy_str}], Avg Time: {avg_time:.2f}s")

    
    # Save results
    df = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Use the original filename if updating, otherwise create new one
    if results_files:
        csv_path = os.path.join(RESULTS_DIR, results_files[0])
        print(f"\n{'='*70}")
        print(f"Appending {len(new_schedules)} new results to {results_files[0]}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if normalized:
            csv_path = os.path.join(RESULTS_DIR, f"schedule_tuning_{NUM_GRAPHS}graphs_{n}v_normalized_{timestamp}.csv")
        else:
            csv_path = os.path.join(RESULTS_DIR, f"schedule_tuning_{NUM_GRAPHS}graphs_{n}v_{timestamp}.csv")
        print(f"\n{'='*70}")
        print(f"Creating new results file")
    
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    print(f"{'='*70}")
    
    # Print summary
    print("\nSchedule Tuning Results Summary:")
    print(df.to_string(index=False))
    
    # Find best schedules by feasibility rate
    df_sorted = df.sort_values('feasible_percentage', ascending=False)
    print(f"\n{'='*70}")
    print("Top 5 schedules by feasibility:")
    print(df_sorted[['schedule', 'feasible_instances', 'total_instances', 'feasible_percentage']].head(5).to_string(index=False))
    print(f"{'='*70}")


if __name__ == "__main__":
    run_schedule_tuning_experiment()
