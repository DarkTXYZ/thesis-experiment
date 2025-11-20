# Available Optimizers for ToQUBO

## Currently Installed and Working:

### 1. **PySA (Python Simulated Annealing)** ✅
```julia
using PySA
model = Model(() -> ToQUBO.Optimizer(PySA.Optimizer))
```
- **Pros**: Good quality solutions, works well with constraints
- **Cons**: Slower for large instances
- **Best for**: General purpose, finding feasible solutions

### 2. **QUBODrivers.RandomSampler** ✅
```julia
using QUBODrivers
model = Model(() -> ToQUBO.Optimizer(QUBODrivers.RandomSampler.Optimizer))
```
- **Pros**: Very fast
- **Cons**: Random solutions, often infeasible
- **Best for**: Quick testing, baseline comparisons

### 3. **QUBODrivers.ExactSampler** ✅
```julia
using QUBODrivers
model = Model(() -> ToQUBO.Optimizer(QUBODrivers.ExactSampler.Optimizer))
```
- **Pros**: Finds optimal solution
- **Cons**: Exponential time (only works for small instances, ~20 variables max)
- **Best for**: Small instances, verifying solutions

### 4. **QUBODrivers.IdentitySampler** ✅
```julia
using QUBODrivers
model = Model(() -> ToQUBO.Optimizer(QUBODrivers.IdentitySampler.Optimizer))
```
- **Pros**: No-op sampler for testing
- **Cons**: Returns empty solution
- **Best for**: Testing, debugging

### 5. **MQLib** ✅
```julia
using MQLib
model = Model(() -> ToQUBO.Optimizer(MQLib.Optimizer))
```
- **Pros**: Multiple heuristics available
- **Cons**: **Does NOT work well for constrained problems like MINLA** (returns all-zero solutions)
- **Best for**: Unconstrained Max-Cut and QUBO problems only
- **❌ Not recommended for MINLA**

## Can Be Installed:

### 6. **SimulatedAnnealing**
```bash
julia -e 'using Pkg; Pkg.add("SimulatedAnnealing")'
```
```julia
using SimulatedAnnealing
# May need adapter for ToQUBO
```

### 7. **IsingSolvers**
```bash
julia -e 'using Pkg; Pkg.add("IsingSolvers")'
```
```julia
using IsingSolvers
# May need adapter for ToQUBO
```

### 8. **QuantumAnnealing**
```bash
julia -e 'using Pkg; Pkg.add("QuantumAnnealing")'
```
```julia
using QuantumAnnealing
# Simulates quantum annealing
```

### 9. **DWave.Neal** (requires dwave-neal Python package)
```bash
pip install dwave-neal
```
```julia
using DWave
model = Model(() -> ToQUBO.Optimizer(DWave.Neal.Optimizer))
```
- **Pros**: Industry-standard simulated annealing
- **Cons**: Requires Python setup
- **Best for**: Benchmarking against D-Wave ecosystem

## Usage Tips:

1. **For small instances (n ≤ 20)**: Use ExactSampler
2. **For medium instances (n ≤ 50)**: Use PySA
3. **For large instances (n > 50)**: Use PySA with time limits
4. **For benchmarking**: Compare PySA with DWave.Neal

⚠️ **Note**: MQLib does NOT work for MINLA - it returns all-zero solutions. Use PySA instead.

## Parameter Tuning:

Different optimizers may accept parameters:
```julia
# PySA with custom parameters
set_optimizer_attribute(model, "num_reads", 100)
set_optimizer_attribute(model, "num_sweeps", 1000)
```

Check each optimizer's documentation for available parameters.
