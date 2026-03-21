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
SEED = 42
RESULTS_JSON = os.path.join(RESULTS_DIR, "PIM_tuning_results.json")


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
        """Build composite key used to detect duplicate experiment configs."""
        return (
            result.get('n'),
            result.get('m'),
            result.get('beta_min'),
            result.get('beta_max'),
            result.get('space_type'),
            result.get('annealing_type'),
            result.get('beta_schedule_type'),
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

    def upsert_result_in_state(self, data: Dict, result: Dict, on_conflict: str = 'update') -> str:
        """Upsert one result into in-memory state. Returns inserted/updated/ignored."""
        if on_conflict not in {'update', 'ignore'}:
            raise ValueError("on_conflict must be 'update' or 'ignore'")

        if '_key_to_pos' not in data:
            data['_key_to_pos'] = self._key_index(data['results'])

        key = self._unique_key(result)
        key_to_pos = data['_key_to_pos']
        if key in key_to_pos:
            if on_conflict == 'ignore':
                return 'ignored'
            pos = key_to_pos[key]
            existing_id = data['results'][pos]['id']
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
                'beta_min': float(best_feasible['beta_min']),
                'beta_max': float(best_feasible['beta_max']),
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


def print_result(config_count: int, total_configs: int, space_type: str, annealing_type: str, beta_min: float, beta_max: float,
                 feasible: bool, energy: float, minla_cost, optimal_cost, elapsed: float):
    status = "✓" if feasible else "✗"
    print(f"[{config_count}/{total_configs}] {space_type:9} | annealing={annealing_type} | beta=({beta_min:.2e}, {beta_max:.2e}) | {status} "
          f"E={energy:12.2f} | cost={minla_cost} | optimal_cost={optimal_cost} | {elapsed:.2f}s")


def run_experiment():
    np.random.seed(SEED)
    datasets = read_dataset()
    
    beta_range_min = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    beta_range_max = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    
    beta_range_min_logspace = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1])
    beta_range_max_logspace = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1])
    
    # beta_range_min = np.array([1e-9])
    # beta_range_max = np.array([1e-9])
    
    # beta_range_min_logspace = np.array([-1])
    # beta_range_max_logspace = np.array([0])
    
    space_types = ['linear', 'geometric', 'logspace']
    annealing_types = ['default', 'fixed_Hp']
    # space_types = ['logspace']
    # annealing_types = ['default']
   
    graph_data = datasets[30]['graphs'][0]
    G = nx.Graph()
    G.add_nodes_from(range(graph_data['num_vertices']))
    G.add_edges_from(graph_data['edges'])
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    bqm = minla.generate_bqm_instance(G)
    optimal_cost = graph_data.get('optimal_cost', None)
    
    solver = PathIntegralAnnealingSampler()
    num_sweeps = 1000
    
    configs = []
    for annealing_type in annealing_types:
        for space_type in space_types:
            if space_type == 'logspace':
                beta_min_range = beta_range_min_logspace
                beta_max_range = beta_range_max_logspace
            else:
                beta_min_range = beta_range_min
                beta_max_range = beta_range_max
            
            for beta_min in beta_min_range:
                for beta_max in beta_max_range:
                    if beta_min > beta_max:
                        continue
                    configs.append((space_type, annealing_type, beta_min, beta_max))
    
    total_configs = len(configs)
    db = ResultsManager(RESULTS_JSON)
    write_stats = {'inserted': 0, 'updated': 0, 'ignored': 0}
    processed = 0
    state = db.load_state()
    existing_keys = set(state['_key_to_pos'].keys())

    try:
        for config_count, (space_type, annealing_type, beta_min, beta_max) in enumerate(configs, 1):
            config_key = (
                int(n),
                int(m),
                float(beta_min),
                float(beta_max),
                space_type,
                annealing_type,
                'custom',
            )
            if config_key in existing_keys:
                write_stats['ignored'] += 1
                processed = config_count
                print(
                    f"[{config_count}/{total_configs}] {space_type:9} | annealing={annealing_type} "
                    f"| beta=({beta_min:.2e}, {beta_max:.2e}) | skipped (already in database)"
                )
                continue

            t0 = time.time()

            Hp_field, Hd_field = generate_field(space_type, annealing_type, beta_min, beta_max, num_sweeps)

            sampleset = solver.sample(
                bqm,
                num_reads=10,
                num_sweeps=num_sweeps,
                beta_schedule_type='custom',
                seed=SEED,
                Hp_field=Hp_field,
                Hd_field=Hd_field
            )

            elapsed = time.time() - t0

            best = sampleset.first
            energy = best.energy
            ordering, feasible = decode_solution(best.sample, n)
            minla_cost = minla.calculate_min_linear_arrangement(G, ordering) if feasible else None
            rel_gap = (minla_cost - optimal_cost) / optimal_cost if (feasible and optimal_cost) else None

            row = {
                'n': n,
                'm': m,
                'beta_min': beta_min,
                'beta_max': beta_max,
                'space_type': space_type,
                'annealing_type': annealing_type,
                'beta_schedule_type': 'custom',
                'energy': energy,
                'feasible': feasible,
                'minla_cost': minla_cost,
                'optimal_cost': optimal_cost,
                'relative_gap': rel_gap,
                'time_s': round(elapsed, 3),
            }

            outcome = db.upsert_result_in_state(state, row, on_conflict='update')
            write_stats[outcome] += 1
            db.save_state(state)
            existing_keys.add(config_key)
            processed = config_count

            print_result(config_count, total_configs, space_type, annealing_type, beta_min, beta_max, feasible, energy, minla_cost, optimal_cost, elapsed)
    except KeyboardInterrupt:
        print(f"\nInterrupted at {processed}/{total_configs}. Partial results are already saved.")

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
        print(f"  Beta range: ({best['beta_min']}, {best['beta_max']})")
        print(f"  Energy: {best['energy']:.2f}")
        if best['minla_cost']:
            print(f"  MinLA cost: {best['minla_cost']}")
    
    df = db.get_all_results()
    return df

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