# Quantum-Inspired Optimization for Minimum Linear Arrangement

A comprehensive research framework for solving the **Minimum Linear Arrangement (MinLA)** problem using quantum-inspired optimization techniques. This codebase implements and compares various solvers including exact methods, heuristic baselines, and quantum-inspired annealing approaches formulated as Quadratic Unconstrained Binary Optimization (QUBO) problems.

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Solvers](#solvers)
- [Experiments](#experiments)
- [Results & Analysis](#results--analysis)
- [Known Issues](#known-issues)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## 🎯 Overview

The **Minimum Linear Arrangement (MinLA)** problem is a classic graph optimization problem that seeks to arrange vertices of a graph along a line such that the sum of edge lengths is minimized. This problem has applications in circuit design, graph drawing, task scheduling, and network optimization.

This research explores:
- **QUBO formulations** with different penalty coefficient strategies (exact bounds vs. Lucas bounds)
- **Quantum-inspired solvers** (OpenJij, QWaveSampler, Simulated Bifurcation)
- **Exact solvers** (Dimod ExactSolver, QuboLite)
- **Classical baseline algorithms** (Spectral Sequencing, Successive Augmentation, Local Search)
- Performance comparison across synthetic and real-world graph datasets

## 🧮 Problem Formulation

### MinLA Definition
Given a graph $G = (V, E)$ with $n$ vertices, find a bijective mapping $\pi: V \to \{1, 2, ..., n\}$ that minimizes:

$$
\text{MinLA}(G, \pi) = \sum_{(u,v) \in E} |\pi(u) - \pi(v)|
$$

### QUBO Formulation
We encode the permutation using a thermometer encoding with binary variables $X_{u,k}$ where $X_{u,k} = 1$ indicates vertex $u$ is assigned position $\leq k$.

**Objective Function:**
$$
H_{\text{objective}} = \sum_{(u,v) \in E} \sum_{k=1}^{n} (X_{u,k} + X_{v,k} - 2X_{u,k}X_{v,k})
$$

**Constraints:**
- **Thermometer Constraint:** Ensures $X_{u,1} \geq X_{u,2} \geq ... \geq X_{u,n}$
$$
H_{\text{thermometer}} = \mu_{\text{thermo}} \sum_{u=1}^{n} \sum_{k=1}^{n-1} (1-X_{u,k})X_{u,k+1}
$$

- **Bijective Constraint:** Ensures exactly one vertex per position
$$
H_{\text{bijective}} = \mu_{\text{bijec}} \sum_{k=1}^{n} \left((n-k) - \sum_{u=1}^{n} X_{u,k}\right)^2
$$

**Total Hamiltonian:**
$$
H = H_{\text{objective}} + H_{\text{thermometer}} + H_{\text{bijective}}
$$

### Penalty Coefficient Strategies
- **Exact Bound:** Theoretically guaranteed penalty coefficients ensuring constraint satisfaction
- **Lucas Bound:** Practical penalty coefficients based on Lucas (2014) formulation

## 📁 Project Structure

```
Experiment/
├── analyze.py                    # Result analysis and visualization
├── analyze_real_world.py         # Real-world dataset analysis
├── exact_experiment.py           # Exact solver experiments
├── quantum_experiment.py         # Main quantum-inspired solver experiments
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── Baseline/                     # Classical baseline algorithms
│   ├── spectral_sequencing.py    # Spectral ordering heuristic
│   ├── successive_augmentation.py # Successive augmentation algorithm
│   └── local_search.py           # Local search optimization
│
├── Dataset/                      # Graph datasets and preprocessing
│   ├── exact_dataset/            # Small graphs for exact solver validation
│   ├── quantum_dataset/          # Synthetic graphs for quantum experiments
│   ├── quantum_real_world_dataset/ # Real-world benchmark graphs
│   ├── processed/                # Preprocessed graph files
│   └── raw/                      # Raw dataset sources
│
├── Solver/                       # Solver implementations
│   ├── penalty_coefficients.py   # Penalty coefficient calculations
│   ├── solver_class.py           # Base solver interface
│   ├── ExactSolver/              # Exact QUBO solvers
│   │   ├── DimodExactSolver.py   # D-Wave Dimod exact solver
│   │   └── QuboLiteSolver.py     # QuboLite solver
│   ├── HeuristicSolver/          # Quantum-inspired heuristic solvers
│   │   ├── OpenJijSolver.py      # OpenJij (Simulated Annealing)
│   │   ├── QWSamplerSolver.py    # Quantum Wave Sampler
│   │   └── SBSolver.py           # Simulated Bifurcation
│   └── QUBO++/                   # C++ QUBO solver framework
│
├── Utils/                        # Utility functions
│   └── min_lin_arrangement.py    # MinLA cost calculation
│
└── Results/                      # Experimental results
    ├── quantum_experiment_*/     # Timestamped experiment outputs
    │   ├── aggregated_results.csv    # Summary statistics
    │   ├── detailed_results.csv      # Per-graph results
    │   ├── real_world_results.csv    # Real-world benchmark results
    │   └── solver_times.csv          # Execution time analysis
    └── *.csv                     # Legacy result files
```

## 🚀 Installation

### Prerequisites
- **Python 3.10+** (tested with Python 3.12)
- **C++ compiler** (for QUBO++ solver, optional)
- **macOS/Linux/Windows** (macOS users: see [Known Issues](#known-issues))

### Quick Start

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Experiment
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Fix macOS OpenMP issue (macOS only):**

If you encounter `OMP: Error #15: Initializing libomp.dylib`, set the environment variable before running:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Or add to the beginning of your Python scripts:
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

4. **Optional: Build QUBO++ solver:**
```bash
cd Solver/QUBO++/minla
make
cd ../../..
```

## 💻 Usage

### Running Quantum Experiments

The main quantum experiment script compares quantum-inspired solvers across synthetic and real-world datasets:

```bash
python quantum_experiment.py
```

**Configuration (edit `CONFIG` in `quantum_experiment.py`):**
```python
CONFIG = ExperimentConfig(
    vertex_counts=[6, 8, 11, 13, 15],      # Graph sizes to test
    penalty_methods=['exact', 'lucas'],     # Penalty coefficient methods
    num_reads=100,                          # Number of annealing reads
    seed=42,                                # Random seed
    use_openjij=True,                       # Enable OpenJij solver
    use_qwavesampler=True,                  # Enable QWaveSampler
    use_simulated_bifurcation=True,         # Enable Simulated Bifurcation
    qwavesampler_types=['path'],            # QWaveSampler variants
    use_synthetic_dataset=True,             # Include synthetic graphs
    use_real_world_dataset=True,            # Include real-world graphs
    success_gap_threshold=0.05,             # 5% optimality gap threshold
    verbose=True
)
```

### Running Exact Solver Experiments

For validation on small graphs with guaranteed optimal solutions:

```bash
python exact_experiment.py
```

This uses exact QUBO solvers (DimodExactSolver) on graphs from `Dataset/exact_dataset/`.

### Analyzing Results

View aggregated statistics and comparisons:

```bash
python analyze.py
```

For real-world dataset analysis:

```bash
python analyze_real_world.py
```

### Running Baseline Algorithms

```bash
# Spectral sequencing baseline
python Baseline/spectral_sequencing.py

# Successive augmentation
python Baseline/successive_augmentation.py

# Local search
python Baseline/local_search.py
```

## 🔧 Solvers

### Exact Solvers
- **DimodExactSolver**: D-Wave Dimod's exact QUBO solver (exponential complexity)
- **QuboLiteSolver**: Lightweight exact solver implementation

### Quantum-Inspired Heuristic Solvers
- **OpenJij**: Simulated Quantum Annealing (SQA) and Simulated Annealing (SA)
- **QWaveSampler**: Quantum wave-based optimization with multiple sampler types
  - Path integral sampler
  - Simulated annealing sampler  
  - Steepest descent sampler
- **Simulated Bifurcation**: Physics-inspired optimization algorithm

### Classical Baselines
- **Spectral Sequencing**: Eigenvector-based graph ordering
- **Successive Augmentation**: Greedy vertex insertion heuristic
- **Local Search**: Iterative improvement through vertex swaps

## 🧪 Experiments

### Experiment Types

1. **Synthetic Dataset Experiments** (`quantum_experiment.py`)
   - Erdős-Rényi random graphs
   - Controlled vertex counts: 6, 8, 11, 13, 15
   - Fixed edge density: 0.5
   - 50 graphs per configuration

2. **Real-World Benchmark Experiments**
   - ENZYMES protein structure graphs
   - PROTEINS interaction graphs
   - ISCAS circuit graphs
   - Social network graphs

3. **Exact Solver Validation** (`exact_experiment.py`)
   - Small graphs (≤8 vertices)
   - Validates QUBO formulation correctness
   - Tests penalty coefficient strategies

### Metrics Evaluated

- **Feasibility Rate**: Percentage of solutions satisfying all constraints
- **Success Rate**: Percentage of solutions within optimality gap threshold
- **Dominance Score**: Normalized ranking across all test instances
- **Relative Gap**: $\frac{\text{solver\_cost} - \text{best\_known}}{\text{best\_known}} \times 100\%$
- **Execution Time**: Wall-clock time per graph
- **Consistency**: Standard deviation of relative gaps

## 📊 Results & Analysis

### Output Files

Each experiment run creates a timestamped directory in `Results/`:

```
Results/quantum_experiment_YYYYMMDD_HHMMSS/
├── aggregated_results.csv    # Summary by solver/dataset/penalty
├── detailed_results.csv       # Individual graph results
├── real_world_results.csv     # Real-world benchmark results
└── solver_times.csv           # Execution time statistics
```

### Result Columns

**Aggregated Results:**
- `solver_name`, `sampler_type`: Solver identification
- `dataset_name`, `num_vertices`: Dataset characteristics
- `penalty_mode`: 'exact' or 'lucas'
- `feasibility_rate`, `success_rate`: Performance metrics
- `avg_relative_gap`, `std_relative_gap`: Optimality measures
- `total_time`: Cumulative execution time

**Detailed Results:**
- Per-graph solver performance
- Energy values and constraint violations
- MinLA costs and feasibility status

### Visualization

Run analysis scripts to generate comparative tables and statistics:

```bash
python analyze.py              # Synthetic dataset analysis
python analyze_real_world.py   # Real-world dataset analysis
```

## ⚠️ Known Issues

### macOS OpenMP Library Conflict

**Error:** `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`

**Cause:** Multiple OpenMP runtimes loaded (common with OpenJij on macOS)

**Solution:**
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

Add this **before** any imports in your scripts, or set in terminal:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

**Note:** This is a workaround. For production use, resolve library conflicts properly.

## 📦 Dependencies

### Core Libraries
- **NetworkX** (3.6.1): Graph manipulation and algorithms
- **NumPy** (2.3.5), **SciPy** (1.15.3): Numerical computing
- **Pandas** (2.3.3): Data analysis and CSV handling
- **Matplotlib** (3.10.8): Visualization

### Quantum-Inspired Solvers
- **OpenJij** (0.11.6): Simulated Quantum/Simulated Annealing
- **PyQUBO** (1.5.0): QUBO problem modeling
- **Simulated-Bifurcation** (2.0.0): Physics-inspired solver
- **QuboLite** (0.8.5): Lightweight QUBO solver

### Exact Solvers
- **Dimod** (0.12.21): D-Wave's QUBO/Ising framework
- **dwave-samplers** (1.7.0): D-Wave sampler implementations
- **dwave-neal** (0.6.0): Simulated annealing sampler

### Optimization & ML
- **scikit-learn** (1.8.0): Machine learning utilities
- **PyTorch** (2.9.1): Deep learning framework (for some solvers)
- **Gurobi** (13.0.0, optional): Commercial optimization solver

### Utilities
- **tqdm** (4.67.1): Progress bars
- **igraph** (1.0.0): High-performance graph library
- **joblib** (1.5.3): Parallel computing

See [requirements.txt](requirements.txt) for complete dependency list with versions.

## 🤝 Contributing

This is a research codebase for thesis work. For questions or collaborations, please contact the repository maintainer.

### Code Structure Guidelines
- Solvers implement the `BaseSolver` interface from `Solver/solver_class.py`
- Use dataclasses for structured result storage
- Follow existing naming conventions for consistency
- Add docstrings to new functions and classes

## 📄 License

[Specify license here]

## 📚 References

- Lucas, A. (2014). "Ising formulations of many NP problems." *Frontiers in Physics*, 2, 5.
- [Add other relevant papers and references]

## 🔗 Related Work

- [Add links to related repositories, papers, or resources]

---

**Last Updated:** January 31, 2026  
**Python Version:** 3.12+  
**Status:** Active Research
