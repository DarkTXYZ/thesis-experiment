import os
import sys
import pickle
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
from dwave.samplers import PathIntegralAnnealingSampler
from typing import Dict, Tuple, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset")
RESULTS_DIR = os.path.join(PARENT_DIR, "Results")
SEEDS = [42, 123, 456, 789, 999]  # 5 different seeds
RESULTS_JSON = os.path.join(RESULTS_DIR, "PIM_tuning_fixed_temperature_results.json")
BEST_RESULTS_JSON = os.path.join(RESULTS_DIR, "PIM_tuning_fixed_temperature_best_results.json")


class ResultsManager:
    """JSON-based results manager for add, remove, adjust operations"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        self._ensure_file()
    
    def _ensure_file(self):
        """Create JSON file if it doesn't exist"""
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w') as f:
                json.dump({'results': [], 'next_id': 1}, f, indent=2)
        else:
            # Verify file is valid JSON, reset if corrupted
            try:
                with open(self.json_path, 'r') as f:
                    json.load(f)
            except (json.JSONDecodeError, ValueError):
                with open(self.json_path, 'w') as f:
                    json.dump({'results': [], 'next_id': 1}, f, indent=2)
    
    def _load(self) -> Dict:
        """Load all data from JSON with error recovery"""
        try:
            with open(self.json_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # File is corrupted, reset it
            with open(self.json_path, 'w') as f:
                json.dump({'results': [], 'next_id': 1}, f, indent=2)
            return {'results': [], 'next_id': 1}
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
    
    def _save(self, data: Dict):
        """Save all data to JSON"""
        data = self._convert_to_serializable(data)
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_state(self) -> Dict:
        """Load mutable state once for repeated operations."""
        data = self._load()
        data['_key_to_pos'] = self._key_index(data['results'])
        return data

    def save_state(self, data: Dict):
        """Persist mutable state to disk."""
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        self._save(data)

    def _unique_key(self, result: Dict) -> Tuple:
        """Build composite key used to detect duplicate experiment configs (without seed)."""
        return (
            result.get('n'),
            result.get('m'),
            result.get('beta'),
            result.get('space_type'),
            result.get('annealing_type'),
            result.get('beta_schedule_type'),
            result.get('bqm_is_normalized'),
        )

    def _key_index(self, results: List[Dict]) -> Dict[Tuple, int]:
        """Map composite keys to row index in stored results list."""
        index = {}
        for i, existing in enumerate(results):
            index[self._unique_key(existing)] = i
        return index

    def get_existing_keys(self) -> set:
        """Get all existing composite config keys from storage."""
        data = self._load()
        return set(self._key_index(data['results']).keys())

    def _is_result_better(self, new_result: Dict, existing_result: Dict) -> bool:
        """Compare if new result is better than existing result.
        Criteria: feasible > not feasible, then lower energy > higher energy"""
        new_feasible = new_result.get('feasible', False)
        existing_feasible = existing_result.get('feasible', False)
        
        # If new is feasible and existing is not, new is better
        if new_feasible and not existing_feasible:
            return True
        # If both have same feasibility, compare energy
        if new_feasible == existing_feasible:
            new_energy = new_result.get('energy', float('inf'))
            existing_energy = existing_result.get('energy', float('inf'))
            return new_energy < existing_energy
        # If new is not feasible but existing is, new is worse
        return False

    def upsert_result_in_state(self, data: Dict, result: Dict, on_conflict: str = 'update') -> str:
        """Upsert one result into in-memory state. Returns inserted/updated/ignored."""
        if on_conflict not in {'update', 'ignore', 'better'}:
            raise ValueError("on_conflict must be 'update', 'ignore', or 'better'")

        if '_key_to_pos' not in data:
            data['_key_to_pos'] = self._key_index(data['results'])

        key = self._unique_key(result)
        key_to_pos = data['_key_to_pos']
        if key in key_to_pos:
            if on_conflict == 'ignore':
                return 'ignored'
            pos = key_to_pos[key]
            existing_result = data['results'][pos]
            
            if on_conflict == 'better':
                # Only update if new result is better than existing
                if not self._is_result_better(result, existing_result):
                    return 'ignored'
            
            # Update the result
            existing_id = existing_result['id']
            data['results'][pos] = {'id': existing_id, **result}
            return 'updated'

        result_with_id = {'id': data['next_id'], **result}
        data['results'].append(result_with_id)
        key_to_pos[key] = len(data['results']) - 1
        data['next_id'] += 1
        return 'inserted'
    
    def add_results_batch(self, results: List[Dict], on_conflict: str = 'update') -> Dict[str, int]:
        """Add multiple results. Returns counts for inserted/updated/ignored."""
        data = self.load_state()
        inserted = 0
        updated = 0
        ignored = 0

        for result in results:
            outcome = self.upsert_result_in_state(data, result, on_conflict=on_conflict)
            if outcome == 'inserted':
                inserted += 1
            elif outcome == 'updated':
                updated += 1
            else:
                ignored += 1

        self.save_state(data)
        return {'inserted': inserted, 'updated': updated, 'ignored': ignored}
    
    def get_all_results(self) -> pd.DataFrame:
        """Retrieve all results as DataFrame"""
        data = self._load()
        df = pd.DataFrame(data['results'])
        return df if len(df) > 0 else pd.DataFrame()
    
    def get_results_filtered(self, space_type: str = None, 
                            annealing_type: str = None,
                            feasible_only: bool = False) -> pd.DataFrame:
        """Get filtered results"""
        data = self._load()
        results = data['results']
        
        if space_type:
            results = [r for r in results if r.get('space_type') == space_type]
        if annealing_type:
            results = [r for r in results if r.get('annealing_type') == annealing_type]
        if feasible_only:
            results = [r for r in results if r.get('feasible')]
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def remove_result(self, result_id: int) -> bool:
        """Remove a single result by ID"""
        data = self._load()
        initial_count = len(data['results'])
        data['results'] = [r for r in data['results'] if r['id'] != result_id]
        self._save(data)
        return len(data['results']) < initial_count
    
    def clear_all_results(self) -> int:
        """Clear all results. Returns count of deleted."""
        data = self._load()
        deleted = len(data['results'])
        data['results'] = []
        self._save(data)
        return deleted
    
    def update_result(self, result_id: int, updates: Dict) -> bool:
        """Update fields in a result"""
        allowed_fields = {'energy', 'feasible', 'minla_cost', 'optimal_cost', 'relative_gap', 'time_s'}
        updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        data = self._load()
        for result in data['results']:
            if result['id'] == result_id:
                result.update(updates)
                self._save(data)
                return True
        return False
    
    def export_to_csv(self, csv_path: str = None) -> str:
        """Export all results to CSV"""
        if csv_path is None:
            csv_path = os.path.join(RESULTS_DIR, "PIM_tuning_experiment.csv")
        
        df = self.get_all_results()
        if len(df) > 0:
            df.to_csv(csv_path, index=False)
        return csv_path
    
    def get_statistics(self) -> Dict:
        """Get summary statistics"""
        df = self.get_all_results()
        
        if len(df) == 0:
            return {'message': 'No results in database'}
        
        feasible_count = df['feasible'].sum()
        stats = {
            'total_results': len(df),
            'feasible_count': int(feasible_count),
            'feasible_percentage': round(feasible_count / len(df) * 100, 1),
            'best_energy': float(df['energy'].min()),
            'avg_energy': round(float(df['energy'].mean()), 2),
            'avg_time_s': round(float(df['time_s'].mean()), 3)
        }
        
        if feasible_count > 0:
            best_feasible = df[df['feasible'] == 1].nsmallest(1, 'energy').iloc[0]
            stats['best_feasible_config'] = {
                'space_type': best_feasible['space_type'],
                'annealing_type': best_feasible['annealing_type'],
                'beta': float(best_feasible['beta']),
                'energy': float(best_feasible['energy']),
                'minla_cost': int(best_feasible['minla_cost']) if pd.notna(best_feasible['minla_cost']) else None
            }
        
        return stats


def read_dataset():
    datasets = {}
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".pkl"):
            filepath = os.path.join(DATASET_PATH, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                datasets[data['num_vertices']] = data
    return datasets

def decode_solution(raw_sample: Dict, n: int) -> Tuple[np.ndarray, bool]:
    sol = np.zeros((n, n), dtype=int)
    for u in range(n):
        for k in range(n):
            val = raw_sample.get(f'X[{u}][{k}]', 0)
            if val:
                sol[u, k] = 1
    is_feasible = check_feasibility(sol, n)
    ordering = np.sum(sol, axis=1)
    return ordering, is_feasible


def check_feasibility(sol: np.ndarray, n: int) -> bool:
    for u in range(n):
        if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
            return False
    labels = np.sum(sol, axis=1)
    return len(np.unique(labels)) == n and np.all(labels > 0) and np.all(labels <= n)


def generate_field(space_type: str, annealing_type: str, beta_min: float, beta_max: float, num_sweeps: int) -> Tuple[np.ndarray, np.ndarray]:
    if annealing_type == 'default':
        if space_type == 'linear':
            Hp_field = np.linspace(beta_min, beta_max, num=num_sweeps)
            Hd_field = np.linspace(beta_max, beta_min, num=num_sweeps)
        elif space_type == 'geometric':
            Hp_field = np.geomspace(beta_min, beta_max, num=num_sweeps)
            Hd_field = np.geomspace(beta_max, beta_min, num=num_sweeps)
        else:
            Hp_field = np.logspace(beta_min, beta_max, num=num_sweeps)
            Hd_field = np.logspace(beta_max, beta_min, num=num_sweeps)
    elif annealing_type == 'fixed_Hp':
        if space_type == 'linear':
            Hp_field = np.linspace(beta_max, beta_max, num=num_sweeps)
            Hd_field = np.linspace(beta_max, beta_min, num=num_sweeps)
        elif space_type == 'geometric':
            Hp_field = np.geomspace(beta_max, beta_max, num=num_sweeps)
            Hd_field = np.geomspace(beta_max, beta_min, num=num_sweeps)
        else:
            Hp_field = np.logspace(beta_max, beta_max, num=num_sweeps)
            Hd_field = np.logspace(beta_max, beta_min, num=num_sweeps)
    return Hp_field, Hd_field


def print_result(config_count: int, total_configs: int, normalized: bool, space_type: str, annealing_type: str, beta: float,
                 feasible: bool, energy: float, minla_cost, optimal_cost, elapsed: float):
    status = "✓" if feasible else "✗"
    print(f"[{config_count}/{total_configs}] normalized={normalized} | {space_type:9} | annealing={annealing_type} | beta={beta:.2e} | {status} "
          f"E={energy:12.2f} | cost={minla_cost} | optimal_cost={optimal_cost} | {elapsed:.2f}s")


def run_experiment():
    datasets = read_dataset()
    
    betas = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    betas_logspace = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1])
    
    space_types = ['linear', 'geometric', 'logspace']
    annealing_types = ['default', 'fixed_Hp']
    normalized = [True, False]
   
    graph_data = datasets[25]['graphs'][0]
    G = nx.Graph()
    G.add_nodes_from(range(graph_data['num_vertices']))
    G.add_edges_from(graph_data['edges'])
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    solver = PathIntegralAnnealingSampler()
    num_sweeps = 1000
    num_reads = 10

    configs = []
    for norm in normalized:
        for annealing_type in annealing_types:
            for space_type in space_types:
                if space_type == 'logspace':
                    beta_chosen = betas_logspace
                else:
                    beta_chosen = betas
                
                for beta in beta_chosen:
                    configs.append((norm, space_type, annealing_type, beta))
    
    total_configs = len(configs)
    num_seeds = len(SEEDS)
    db = ResultsManager(RESULTS_JSON)
    write_stats = {'inserted': 0, 'updated': 0, 'ignored': 0}
    processed = 0
    state = db.load_state()
    existing_keys = set(state['_key_to_pos'].keys())

    try:
        for config_count, (norm, space_type, annealing_type, beta) in enumerate(configs, 1):
            # Build config key without seed
            config_key = (
                int(n),
                int(m),
                float(beta),
                space_type,
                annealing_type,
                'custom',
                norm,
            )
            
            # Skip if this config already exists in DB
            if config_key in existing_keys:
                write_stats['ignored'] += num_seeds
                print(f"[{config_count}/{total_configs}] Config already exists (all {num_seeds} seeds), skipping...")
                continue
            
            # Run each configuration with all 5 seeds and collect results
            seed_results = []
            
            for seed_idx, seed in enumerate(SEEDS, 1):
                bqm = minla.generate_bqm_instance(G)
                if norm:
                    bqm.normalize()
                optimal_cost = graph_data.get('optimal_cost', None)

                t0 = time.time()

                Hp_field, Hd_field = generate_field(space_type, annealing_type, beta, beta, num_sweeps)

                sampleset = solver.sample(
                    bqm,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps,
                    beta_schedule_type='custom',
                    seed=seed,
                    Hp_field=Hp_field,
                    Hd_field=Hd_field
                )

                elapsed = time.time() - t0

                best = sampleset.first
                energy = best.energy
                ordering, feasible = decode_solution(best.sample, n)
                minla_cost = minla.calculate_min_linear_arrangement(G, ordering) if feasible else None
                rel_gap = (minla_cost - optimal_cost) / optimal_cost if (feasible and optimal_cost) else None

                seed_result = {
                    'n': n,
                    'm': m,
                    'beta': beta,
                    'bqm_is_normalized': norm,
                    'space_type': space_type,
                    'annealing_type': annealing_type,
                    'beta_schedule_type': 'custom',
                    'energy': energy,
                    'feasible': feasible,
                    'minla_cost': minla_cost,
                    'optimal_cost': optimal_cost,
                    'relative_gap': rel_gap,
                    'time_s': round(elapsed, 3),
                    'seed_used': seed,  # Track which seed gave the best result, but not in the key
                }
                seed_results.append(seed_result)
                
                print_result(config_count * num_seeds, total_configs * num_seeds, norm, space_type, annealing_type, beta, feasible, energy, minla_cost, optimal_cost, elapsed)
                print(f"    └─ Seed {seed} ({seed_idx}/{num_seeds})")
            
            # Select best result from all seeds
            best_result = max(seed_results, key=lambda x: (x['feasible'], -x['energy']))
            
            outcome = db.upsert_result_in_state(state, best_result, on_conflict='better')
            write_stats[outcome] += 1
            db.save_state(state)
            existing_keys.add(config_key)
            processed = config_count
            
    except KeyboardInterrupt:
        print(f"\nInterrupted at config {processed}/{total_configs}. Partial results are already saved.")

    print(
        f"\n✓ Database write complete | inserted={write_stats['inserted']}, "
        f"updated={write_stats['updated']}, ignored={write_stats['ignored']}"
    )
    
    csv_path = db.export_to_csv()
    print(f"✓ Results exported to {csv_path}")
    
    stats = db.get_statistics()
    
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"Total configurations tested: {stats['total_results']}")
    print(f"Feasible solutions: {stats['feasible_count']} ({stats['feasible_percentage']}%)")
    print(f"Best energy: {stats['best_energy']:.2f}")
    print(f"Average energy: {stats['avg_energy']}")
    print(f"Average time: {stats['avg_time_s']}s")
    
    if 'best_feasible_config' in stats:
        best = stats['best_feasible_config']
        print(f"\nLowest energy feasible solution:")
        print(f"  Space type: {best['space_type']}")
        print(f"  Annealing type: {best['annealing_type']}")
        print(f"  Betas: {best['beta']}")
        print(f"  Energy: {best['energy']:.2f}")
        if best['minla_cost']:
            print(f"  MinLA cost: {best['minla_cost']}")
    
    df = db.get_all_results()
    
    # Select best results from each configuration across all seeds
    print("\n" + "=" * 70)
    print(" SELECTING BEST RESULTS ACROSS SEEDS")
    print("=" * 70)
    best_df = select_best_results_per_config()
    print(f"✓ Best results selected and saved to {BEST_RESULTS_JSON}")
    
    return df


def select_best_results_per_config() -> pd.DataFrame:
    """Select the best result from each configuration across all seeds"""
    db = ResultsManager(RESULTS_JSON)
    df = db.get_all_results()
    
    if len(df) == 0:
        print("No results to select from")
        return pd.DataFrame()
    
    # Group by configuration (excluding seed)
    config_cols = ['n', 'm', 'beta', 'bqm_is_normalized', 'space_type', 'annealing_type']
    
    best_results = []
    grouped = df.groupby(config_cols)
    
    for config, group in grouped:
        # For each config, select best based on: feasible first, then lowest energy, then highest minla_cost
        feasible_results = group[group['feasible'] == 1]
        
        if len(feasible_results) > 0:
            best = feasible_results.nsmallest(1, 'energy').iloc[0]
        else:
            # If no feasible, take lowest energy
            best = group.nsmallest(1, 'energy').iloc[0]
        
        best_results.append(best.to_dict())
    
    best_df = pd.DataFrame(best_results)
    
    # Save best results
    best_results_data = {
        'results': best_df.to_dict('records'),
        'summary': {
            'total_unique_configs': len(best_df),
            'total_seeds_per_config': len(SEEDS),
            'feasible_count': int(best_df['feasible'].sum()),
            'feasible_percentage': round(best_df['feasible'].sum() / len(best_df) * 100, 1),
            'best_energy': float(best_df['energy'].min()),
            'avg_energy': round(float(best_df['energy'].mean()), 2),
            'best_minla_cost': int(best_df[best_df['feasible'] == 1]['minla_cost'].min()) if (best_df['feasible'] == 1).any() else None,
        }
    }
    
    with open(BEST_RESULTS_JSON, 'w') as f:
        json.dump(best_results_data, f, indent=2, default=str)
    
    # Print summary
    print(f"\nBest Results Summary (across {len(SEEDS)} seeds per config):")
    print(f"  Total unique configurations: {len(best_df)}")
    print(f"  Feasible solutions: {best_results_data['summary']['feasible_count']} ({best_results_data['summary']['feasible_percentage']}%)")
    print(f"  Best energy: {best_results_data['summary']['best_energy']:.2f}")
    print(f"  Average energy: {best_results_data['summary']['avg_energy']}")
    if best_results_data['summary']['best_minla_cost']:
        print(f"  Best MinLA cost: {best_results_data['summary']['best_minla_cost']}")
    
    return best_df

def view_results(space_type: str = None, annealing_type: str = None, feasible_only: bool = False):
    """View results from database with optional filters"""
    db = ResultsManager(RESULTS_JSON)
    df = db.get_results_filtered(space_type=space_type, annealing_type=annealing_type, feasible_only=feasible_only)
    print(f"\nFound {len(df)} results:")
    print(df.to_string())


def remove_result_by_id(result_id: int):
    """Remove a specific result by ID"""
    db = ResultsManager(RESULTS_JSON)
    if db.remove_result(result_id):
        print(f"✓ Result {result_id} removed")
    else:
        print(f"✗ Result {result_id} not found")


def adjust_result(result_id: int, **updates):
    """Adjust/update a result (e.g., adjust_result(5, energy=100.5, feasible=True))"""
    db = ResultsManager(RESULTS_JSON)
    if db.update_result(result_id, updates):
        print(f"✓ Result {result_id} updated with: {updates}")
    else:
        print(f"✗ Result {result_id} not found or invalid fields")


def clear_results():
    """Clear all results from database"""
    db = ResultsManager(RESULTS_JSON)
    count = db.clear_all_results()
    print(f"✓ Cleared {count} results from database")


def get_stats():
    """Get database statistics"""
    db = ResultsManager(RESULTS_JSON)
    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    df = run_experiment()