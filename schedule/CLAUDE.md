# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python project implementing a quantum computing task scheduling algorithm. The goal is to optimize task scheduling on a quantum computer topology to minimize total runtime while respecting quantum bit connectivity constraints.

## Development Environment

- **Python Version**: 3.13
- **Package Manager**: uv
- **Core Dependencies**: mcp[cli]>=1.13.1

## Common Commands

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>

# Run the application
uv run python main.py
```

### Running Tests
Since no test framework is configured yet, run individual test files with:
```bash
uv run python -m pytest <test_file.py>
# or
uv run python <test_file.py>
```

## Project Structure

- `main.py` - Entry point (currently placeholder)
- `a.md` - Requirements specification for quantum scheduling algorithm
- `pyproject.toml` - Project configuration
- `uv.lock` - Dependency lock file

## Algorithm Requirements

Based on `a.md`, implement a quantum task scheduler with:
- **Topology**: Quantum bits as nodes, edges for coupling
- **Tasks**: Attributes `k` (qubits needed), `d` (duration), and task topology structure
- **Constraints**:
  - Tasks can run in parallel if resources available
  - All tasks submitted simultaneously (no arrival times)
  - k qubits must be topologically connected
  - No overlapping qubit usage during execution
  - **NEW**: SWAP cost calculation for topology embedding
- **Objective**: Minimize total runtime including SWAP overhead

## SWAP Cost Calculation

For each task with logical topology \(G_{prog}=(V_p,E_p)\) and physical topology \(G_{chip}=(V_c,E_c)\):

1. **SWAP count**: \(N_{swap} \approx \sum_{(u,v)\in E_p} (\mathrm{dist}_{G_{chip}}(\pi(u),\pi(v)) - 1)\)
2. **Adjusted duration**: \(d'(T) \approx d(T) + c_{swap} \cdot N_{swap}\)

Where \(c_{swap}\) is the equivalent time for a single SWAP operation.

## Architecture Notes

The scheduling algorithm should:
1. Model quantum computer topology as a graph
2. Handle task topology embedding into physical topology
3. Calculate SWAP costs for non-perfect embeddings
4. Implement task scheduling with connectivity and SWAP constraints
5. Optimize for parallel execution while respecting resource limits

## Next Steps

1. Define quantum topology data structures
2. Implement task scheduling algorithm
3. Add visualization for schedules
4. Create test cases with sample topologies and tasks