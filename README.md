# Experiment

Research experiment codebase for thesis work on Minimum Linear Arrangement problem.

## Structure

- **Baseline/** - Baseline algorithms (Gurobi, Spectral Sequencing, Successive Augmentation)
- **Dataset/** - Graph dataset processing and generation tools
- **Results/** - Experimental results in CSV format
- **Solver/** - Various solver implementations (Gurobi, PySA.jl, QUBO++)
- **Utils/** - Utility functions for minimum linear arrangement

## Setup

### Prerequisites
- Python 3.x
- Julia (for PySA.jl solver)
- C++ compiler (for QUBO++ solver)
- Gurobi (optional, for optimization solver)

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# For QUBO++ solver
cd Solver/QUBO++/minla
make
```

## Usage

See individual directories for specific usage instructions.

## Results

Experimental results are stored in the `Results/` directory in CSV format.
