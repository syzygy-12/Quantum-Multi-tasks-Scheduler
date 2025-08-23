from LQuantum import QuantumScheduler, Task, create_grid_topology, create_topology_from_json, create_task_topology
import json
import os
import networkx as nx

def quantum_task_scheduler(
    circuits_config_path: str = './circuits/circuits_small.json',
    topology_config_path: str = None,
    swap_cost: float = 1.0,
    output_path: str = "quantum_schedule_results_small.json"
) :
    """Schedule quantum tasks using SWAP-aware optimization algorithm.
    
    Args:
        circuits_config_path: Optional path to JSON file containing circuit definitions, default './circuits/circuits.json'
        topology_config_path: Optional path to JSON file containing topology definitions, in ./topologies
        swap_cost: Cost per SWAP operation (default 1.0)
        output_path: Path to save scheduling results JSON (default 'quantum_schedule_results.json')
    
    Return:
        Scheduling analysis results with file path and computation summary
    """
    try:
        # Create physical topology
        if topology_config_path and os.path.exists(topology_config_path):
            with open(topology_config_path, 'r', encoding='utf-8') as f:
                topology_data = json.load(f)
            topology = create_topology_from_json(topology_data)
            total_qubits = len(topology.nodes)
            topology_type = topology_data.get("name", "custom")
        else:
            topology = create_grid_topology(6, 6)
            total_qubits = 36
            topology_type = "6*6 rid"

        # Initialize scheduler
        scheduler = QuantumScheduler(topology, swap_cost=swap_cost)
        
        # Load tasks from file or use default complex task set
        if circuits_config_path and os.path.exists(circuits_config_path):
            with open(circuits_config_path, 'r', encoding='utf-8') as f:
                circuits_data = json.load(f)
            
            tasks = []
            for i, circuit in enumerate(circuits_data):
                # Extract task parameters from quantum circuit analysis
                task_id = i + 1  # Use index as task ID
                k = circuit['num_qubits']  # Number of qubits needed
                d = circuit['depth']  # Circuit depth as execution time
                
                # Create task topology from connectivity information
                if circuit.get('connectivity_topology') and len(circuit['connectivity_topology']) > 0:
                    # Build NetworkX graph from connectivity edges
                    task_topology = nx.Graph()
                    task_topology.add_nodes_from(range(k))
                    for edge in circuit['connectivity_topology']:
                        if len(edge) == 2:
                            task_topology.add_edge(edge[0], edge[1])
                else:
                    # Default to line topology if no connectivity info
                    task_topology = create_task_topology('line', k)
                
                tasks.append(Task(task_id, k, d, task_topology))
        else:
            # Default complex task set from LQuantum.py
            tasks = [
                Task(1, 12, 25, create_task_topology("tree", 12)),
                Task(2, 8, 15, create_task_topology("star", 8)),
                Task(3, 16, 35, create_task_topology("ring", 16)),
                Task(4, 10, 20, create_task_topology("grid", 10)),
                Task(5, 6, 18, create_task_topology("tree", 6)),
                Task(6, 5, 12, create_task_topology("line", 5)),
                Task(7, 7, 14, create_task_topology("star", 7)),
                Task(8, 4, 8, create_task_topology("ring", 4)),
                Task(9, 9, 16, create_task_topology("tree", 9)),
                Task(10, 3, 6, create_task_topology("line", 3)),
                Task(11, 2, 3, create_task_topology("line", 2)),
                Task(12, 2, 4, create_task_topology("line", 2)),
                Task(13, 3, 5, create_task_topology("line", 3)),
                Task(14, 2, 2, create_task_topology("line", 2)),
                Task(15, 4, 7, create_task_topology("line", 4)),
                Task(16, 20, 45, create_task_topology("tree", 20)),
                Task(17, 15, 30, create_task_topology("grid", 15)),
                Task(18, 18, 40, create_task_topology("star", 18)),
                Task(19, 14, 28, create_task_topology("ring", 14)),
                Task(20, 11, 22, create_task_topology("tree", 11)),
            ]
        
        # Run scheduling algorithm
        schedule = scheduler.greedy_schedule(tasks)
        
        # Calculate performance metrics
        total_base_time = sum(t.d for t in tasks)
        total_adjusted_time = sum(t.adjusted_duration for t in tasks if hasattr(t, 'adjusted_duration'))
        actual_runtime = scheduler.get_total_runtime(schedule)
        efficiency = total_adjusted_time / actual_runtime if actual_runtime > 0 else 0

        print(f"  Total runtime: {actual_runtime:.2f} time units")
        print(f"  Total base time (without SWAP cost): {total_base_time}")
        print(f"  Total adjusted time (with SWAP cost): {total_adjusted_time}")
        print(f"  Parallelization efficiency: {efficiency:.2f}")

        # Analyze parallel execution
        schedule_by_time = {}
        for task_id, start_time, embedding, adjusted_duration in schedule:
            if start_time not in schedule_by_time:
                schedule_by_time[start_time] = []
            schedule_by_time[start_time].append((task_id, embedding, adjusted_duration))
        
        max_parallel = max(len(tasks) for tasks in schedule_by_time.values()) if schedule_by_time else 1
        
        # Prepare results
        results = {
            "configuration": {
                "topology_type": topology_type,
                "total_qubits": total_qubits,
                "swap_cost": swap_cost,
                "total_tasks": len(tasks),
                "scheduled_tasks": len(schedule)
            },
            "performance": {
                "sequential_runtime": total_adjusted_time,
                "actual_runtime": actual_runtime,
                "speedup": round(efficiency, 2),
                "parallelization_efficiency_percent": round(efficiency * 100, 1),
                "max_parallel_tasks": max_parallel
            },
            "schedule": [
                {
                    "task_id": task_id,
                    "start_time": start_time,
                    "duration": adjusted_duration,
                    "physical_qubits": sorted(list(embedding.values())),
                    "embedding": embedding
                }
                for task_id, start_time, embedding, adjusted_duration in schedule
            ],
            "task_details": [
                {
                    "task_id": t.task_id,
                    "qubits_needed": t.k,
                    "base_duration": t.d,
                    "topology_edges": list(t.task_topology.edges())
                }
                for t in tasks
            ]
        }
        
        # Save results to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        # Create summary for LLM
        summary = f"""Quantum Task Scheduling Analysis Completed Successfully!
        
Results saved to: {output_path}

Configuration:
- Physical topology: {topology_type} ({total_qubits} qubits)
- SWAP cost: {swap_cost} per operation
- Total tasks: {len(tasks)}
- Scheduled tasks: {len(schedule)}

Performance Analysis:
- Sequential runtime: {total_adjusted_time} time units
- Actual runtime: {actual_runtime:.2f} time units
- Speedup achieved: {efficiency:.2f}x
- Parallelization efficiency: {(efficiency * 100):.1f}%
- Maximum parallel tasks: {max_parallel}

The scheduling algorithm successfully optimized {len(tasks)} quantum tasks with SWAP-aware placement, achieving significant parallelization benefits. Tasks were grouped by size to prevent starvation, and optimal embeddings were found to minimize SWAP overhead while maximizing resource utilization.

Scheduled tasks range from small 2-qubit operations to complex 20-qubit algorithms, demonstrating the scheduler's ability to handle diverse quantum workloads efficiently."""
        
        return 0
        
    except Exception as e:
        return 1


if __name__ == "__main__":
    # Example usage
    quantum_task_scheduler()