# Quantum Task Scheduler with SWAP-aware Optimization

A high-performance quantum task scheduler that optimizes quantum circuit placement on physical quantum computers with topology constraints, including SWAP gate cost calculation and parallel execution optimization.



![ibm_rochester](.\ibm_rochester.gif)



![mesh66](.\mesh66.gif)



## Features

- **SWAP-aware scheduling**: Optimizes placement considering SWAP gate overhead
- **Dense topology embedding**: Finds connected subgraphs for quantum circuit placement
- **Parallel execution**: Maximizes resource utilization through intelligent batching
- **Multiple topology support**: Grid, line, star, ring, and tree topologies
- **MCP integration**: Natural language interface for topology creation
- **Performance optimization**: Aggressive search pruning for real-time scheduling

## Quick Start

### Basic Usage

```python
from main import QuantumScheduler, Task, create_grid_topology

# Create quantum computer topology
topology = create_grid_topology(6, 6)  # 6x6 grid with 36 qubits
scheduler = QuantumScheduler(topology, swap_cost=1.0)

# Define quantum tasks
tasks = [
    Task(1, 8, 15, create_task_topology("star", 8)),
    Task(2, 12, 25, create_task_topology("tree", 12)),
]

# Run scheduling
schedule = scheduler.greedy_schedule(tasks)
```

### Using MCP Tools

The system includes MCP (Model Context Protocol) tools for natural language interaction:

#### Create Topologies
```bash
# Create a custom topology via MCP
create_topology(nodes=[0,1,2,3,4], edges=[[0,1],[1,2],[2,3],[3,4]], name="linear_5")
```

#### Schedule Tasks
```bash
# Schedule quantum tasks with full analysis
quantum_task_scheduler(topology_config_path="./topologies/grid.json")
```

## Core Components

### QuantumScheduler Class
- **find_embeddings()**: Fast embedding discovery with early termination
- **calculate_swap_cost()**: Computes SWAP gate overhead for embeddings
- **greedy_schedule()**: Multi-batch scheduling to prevent starvation
- **is_dense_embedding()**: Validates connected subgraph requirements

### Task Class
- **task_id**: Unique identifier
- **k**: Number of qubits required
- **d**: Base execution duration
- **task_topology**: Logical circuit connectivity (NetworkX graph)

## Topology Creation

### Built-in Topologies
```python
from main import create_grid_topology, create_line_topology

# Grid topology (36 qubits)
grid = create_grid_topology(6, 6)

# Line topology (10 qubits)
line = create_line_topology(10)
```

### Custom Topologies via MCP
```bash
# Pentagon topology
create_topology([0,1,2,3,4], [[0,1],[1,2],[2,3],[3,4],[4,0]], "pentagon")

# Fully connected triangle
create_topology([0,1,2], [[0,1],[1,2],[2,0]], "triangle")
```

## Performance Features

### Optimization Strategies
1. **Early termination**: Stops after finding good solutions
2. **Limited branching**: Explores only 2 neighbors per node
3. **Promising starts**: Focuses on high-connectivity qubits
4. **Batch processing**: Groups tasks by size to prevent starvation

### Performance Metrics
- **Speedup**: Achieved parallelization vs sequential execution
- **SWAP overhead**: Additional cost from topology embedding
- **Resource utilization**: Percentage of qubits used simultaneously

## Example Output

```json
{
  "configuration": {
    "topology_type": "6*6 grid",
    "total_qubits": 36,
    "swap_cost": 1.0,
    "total_tasks": 20
  },
  "performance": {
    "sequential_runtime": 400,
    "actual_runtime": 85.5,
    "speedup": 4.68,
    "parallelization_efficiency": 468.0,
    "max_parallel_tasks": 6
  }
}
```

## Advanced Usage

### Custom Circuit Analysis
```python
# Analyze quantum circuits from directory
analyze_quantum_directory(
    input_dir="./quantum_circuits",
    output_file="./analysis_results.json",
    backend="FakeLima"
)
```

### Visualization
```python
# Generate GIF animations of scheduling progress
create_gif_with_custom_order(
    folder_path="./chip_schedules",
    output_name="scheduling_animation.gif"
)
```


## Technical Details

### SWAP Cost Calculation
```
adjusted_duration = base_duration + (swap_cost * total_swaps)
```

### Dense Component Validation
Ensures embedded qubits form a connected induced subgraph with no external nodes in paths.

## Contributing

This is a defensive security-focused quantum scheduler. Contributions for:
- Additional topology types
- Enhanced optimization algorithms  
- Better scheduling heuristics
- Visualization improvements

are welcome.