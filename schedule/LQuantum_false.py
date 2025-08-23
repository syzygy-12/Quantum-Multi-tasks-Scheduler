import networkx as nx
from typing import List, Tuple, Set, Dict, Optional
import itertools
import math


class Task:
    def __init__(self, task_id: int, k: int, d: int, task_topology: nx.Graph):
        self.task_id = task_id
        self.k = k  # qubits needed
        self.d = d  # base duration
        self.task_topology = task_topology  # logical topology
        self.adjusted_duration = d  # will be updated with SWAP costs


class QuantumScheduler:
    def __init__(self, topology: nx.Graph, swap_cost: float = 1.0):
        self.topology = topology
        self.num_qubits = len(topology.nodes())
        self.swap_cost = swap_cost  # cost per SWAP operation
        self.distance_cache = {}  # cache for shortest path distances
        
    def get_distance(self, u: int, v: int) -> int:
        """Get shortest path distance between two qubits."""
        if (u, v) not in self.distance_cache:
            try:
                self.distance_cache[(u, v)] = nx.shortest_path_length(self.topology, u, v)
            except nx.NetworkXNoPath:
                self.distance_cache[(u, v)] = float('inf')
        return self.distance_cache[(u, v)]
    
    def calculate_swap_cost(self, task: Task, embedding: Dict[int, int]) -> float:
        """Calculate SWAP cost for a given task embedding."""
        if len(embedding) != len(task.task_topology.nodes()):
            return float('inf')
        
        total_distance = 0
        for (u, v) in task.task_topology.edges():
            physical_u = embedding[u]
            physical_v = embedding[v]
            distance = self.get_distance(physical_u, physical_v)
            total_distance += max(0, distance - 1)
        
        return total_distance * self.swap_cost
    
    def find_embeddings(self, task: Task, available_qubits: Set[int]) -> List[Tuple[Dict[int, int], float]]:
        """Simple and efficient diffusion-based embedding finder."""
        embeddings = []
        task_nodes = list(task.task_topology.nodes())
        
        for start in available_qubits:
            # 简单的多轮扩散
            current_set = {start}
            
            while len(current_set) < task.k:
                # 找到所有边界节点
                frontier = set()
                for node in current_set:
                    neighbors = set(self.topology.neighbors(node)) & available_qubits - current_set
                    frontier.update(neighbors)
                
                if not frontier:
                    break  # 无法继续扩散
                    
                # 添加一个边界节点（选择连接度最高的）
                next_node = max(frontier, key=lambda x: len(set(self.topology.neighbors(x)) & available_qubits))
                current_set.add(next_node)
            
            if len(current_set) == task.k:
                region_list = sorted(current_set)
                embedding = {task_nodes[i]: region_list[i] for i in range(task.k)}
                swap_cost = self.calculate_swap_cost(task, embedding)
                adjusted_duration = task.d + swap_cost
                embeddings.append((embedding, adjusted_duration))
                
        
        return embeddings

    def is_dense_embedding(self, embedding: Dict[int, int]) -> bool:
        """Check if embedded qubits form a dense component."""
        if not embedding:
            return False

        physical_qubits = set(embedding.values())

        # All embedded qubits must form a connected subgraph
        if len(physical_qubits) > 1:
            subgraph = self.topology.subgraph(list(physical_qubits))
            if not nx.is_connected(subgraph):
                return False

        return True

    def greedy_schedule(self, tasks: List[Task]) -> List[Tuple[int, int, Dict[int, int], float]]:
        """Multi-batch scheduling with pipeline support - prevents starvation of large tasks."""
        qubit_end_time = [0] * self.num_qubits
        schedule = []
        
        # Group tasks by size to prevent starvation
        small_tasks = [t for t in tasks if t.k <= 5]
        medium_tasks = [t for t in tasks if 5 < t.k <= 10]
        large_tasks = [t for t in tasks if t.k > 10]
        
        # Process tasks in batches: large first (to reserve space), then medium, then small
        all_batches = [("Large", large_tasks), ("Medium", medium_tasks), ("Small", small_tasks)]
        
        current_time = 0
        
        for batch_name, batch_tasks in all_batches:
            if not batch_tasks:
                continue
                
            print(f"\nProcessing {batch_name} batch ({len(batch_tasks)} tasks)...")
            
            # Sort tasks within batch by duration (shortest first for fairness)
            batch_tasks = sorted(batch_tasks, key=lambda t: (t.d, t.k))
            
            for task in batch_tasks:
                # Find earliest time when task can be scheduled
                best_start_time = None
                best_embedding = None
                best_duration = float('inf')
                
                # Check multiple time slots for optimal placement
                time_slots = sorted(set([0] + qubit_end_time))
                

                for slot_time in time_slots:
                    # Find available qubits at this time
                    available_qubits = {q for q in range(self.num_qubits) 
                                      if qubit_end_time[q] <= slot_time}

                    if len(available_qubits) < task.k:
                        continue
                    
                    # Find embeddings for this task
                    embeddings = self.find_embeddings(task, available_qubits)
                    
                    
                    if embeddings:
                        # Choose best embedding (minimum SWAP cost)
                        embedding, duration = min(embeddings, key=lambda x: x[1])
                        
                        if slot_time + duration < best_duration:
                            best_start_time = slot_time
                            best_embedding = embedding
                            best_duration = slot_time + duration

                if best_start_time is not None:
                    # Schedule the task
                    physical_qubits = set(best_embedding.values())
                    schedule.append((task.task_id, best_start_time, best_embedding, best_duration - best_start_time))
                    
                    # Update qubit end times
                    for qubit in physical_qubits:
                        qubit_end_time[qubit] = best_start_time + (best_duration - best_start_time)
                else:
                    print(f"  Warning: Cannot schedule Task {task.task_id} (k={task.k}, d={task.d})")
        
        return schedule
    
    def get_total_runtime(self, schedule: List[Tuple[int, int, Dict[int, int], float]]) -> int:
        """Calculate total runtime."""
        if not schedule:
            return 0
        return max(start_time + duration for _, start_time, _, duration in schedule)

def create_topology_from_json(json_data: dict) -> nx.Graph:
    """Create a topology from JSON data."""
    G = nx.Graph()
    nodes = json_data.get("nodes", [])
    edges = json_data.get("edges", [])
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def create_line_topology(n: int) -> nx.Graph:
    """Create a line topology with n qubits."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def create_grid_topology(rows: int, cols: int) -> nx.Graph:
    """Create a 2D grid topology."""
    G = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            G.add_node(node)
            
            # Add edges to neighbors
            if r > 0:
                G.add_edge(node, (r - 1) * cols + c)
            if c > 0:
                G.add_edge(node, r * cols + (c - 1))
            if r < rows - 1:
                G.add_edge(node, (r + 1) * cols + c)
            if c < cols - 1:
                G.add_edge(node, r * cols + (c + 1))
    return G


def create_task_topology(topology_type: str, k: int) -> nx.Graph:
    """Create task topology based on type."""
    if topology_type == "line":
        G = nx.Graph()
        G.add_nodes_from(range(k))
        for i in range(k - 1):
            G.add_edge(i, i + 1)
        return G
    elif topology_type == "star":
        G = nx.Graph()
        G.add_nodes_from(range(k))
        for i in range(1, k):
            G.add_edge(0, i)
        return G
    elif topology_type == "ring":
        G = nx.Graph()
        G.add_nodes_from(range(k))
        for i in range(k):
            G.add_edge(i, (i + 1) % k)
        return G
    elif topology_type == "tree":
        G = nx.Graph()
        G.add_nodes_from(range(k))
        # Create a complete binary tree structure
        for i in range(1, k):
            parent = (i - 1) // 2
            G.add_edge(parent, i)
        return G
    elif topology_type == "grid":
        # Create a 2D grid topology
        import math
        side = int(math.ceil(math.sqrt(k)))
        G = nx.Graph()
        G.add_nodes_from(range(k))
        
        # Add edges for grid topology
        for i in range(k):
            row, col = i // side, i % side
            # Right neighbor
            if col < side - 1 and i + 1 < k:
                G.add_edge(i, i + 1)
            # Bottom neighbor
            if row < side - 1 and i + side < k:
                G.add_edge(i, i + side)
        return G
    else:
        # Default to line topology
        return create_task_topology("line", k)


def main():
    print("Quantum Task Scheduler with SWAP Cost")
    print("=" * 40)

# Create a 6x6 grid topology (36 qubits) for complex task scheduling
    topology = create_grid_topology(6, 6)
    scheduler = QuantumScheduler(topology, swap_cost=1.0)
    
    # Define complex and intense quantum tasks
    tasks = [
        # Large quantum algorithms with complex topologies
        Task(1, 12, 25, create_task_topology("tree", 12)),    # Quantum error correction
        Task(2, 8, 15, create_task_topology("star", 8)),      # Quantum teleportation
        Task(3, 16, 35, create_task_topology("ring", 16)),    # Quantum simulation
        Task(4, 10, 20, create_task_topology("grid", 10)),    # Grover's algorithm
        Task(5, 6, 18, create_task_topology("tree", 6)),      # QAOA optimization
        
        # Medium-sized quantum circuits
        Task(6, 5, 12, create_task_topology("line", 5)),      # Quantum adder
        Task(7, 7, 14, create_task_topology("star", 7)),      # Phase estimation
        Task(8, 4, 8, create_task_topology("ring", 4)),       # Quantum Fourier transform
        Task(9, 9, 16, create_task_topology("tree", 9)),      # Shor's algorithm component
        Task(10, 3, 6, create_task_topology("line", 3)),      # Quantum gates
        
        # Small quantum primitives for parallel execution
        Task(11, 2, 3, create_task_topology("line", 2)),      # CNOT gates
        Task(12, 2, 4, create_task_topology("line", 2)),      # Hadamard networks
        Task(13, 3, 5, create_task_topology("line", 3)),      # Toffoli gates
        Task(14, 2, 2, create_task_topology("line", 2)),      # Measurement circuits
        Task(15, 4, 7, create_task_topology("line", 4)),      # Entanglement generation
        
        # Very complex quantum algorithms
        Task(16, 20, 45, create_task_topology("tree", 20)),   # Surface code
        Task(17, 15, 30, create_task_topology("grid", 15)),   # Topological quantum computing
        Task(18, 18, 40, create_task_topology("star", 18)),   # Quantum machine learning
        Task(19, 14, 28, create_task_topology("ring", 14)),   # Quantum chemistry simulation
        Task(20, 11, 22, create_task_topology("tree", 11)),   # Variational quantum eigensolver
    ]
    
    print("Tasks:")
    for task in tasks:
        print(f"  Task {task.task_id}: k={task.k}, d={task.d}, topology={list(task.task_topology.edges())}")
    
    print(f"\nPhysical topology: 6x6 grid ({len(topology.nodes())} qubits)")
    print(f"SWAP cost per operation: {scheduler.swap_cost}")
    
    # Run SWAP-aware scheduling
    schedule = scheduler.greedy_schedule(tasks)
    
    # Calculate theoretical sequential time
    total_base_time = sum(t.d for t in tasks)
    sequential_time = total_base_time
    
    # Calculate actual time with parallelization
    actual_runtime = scheduler.get_total_runtime(schedule)
    
    # Calculate parallelization efficiency
    efficiency = sequential_time / actual_runtime if actual_runtime > 0 else 0
    
    print(f"\nParallelization Analysis:")
    print(f"  Sequential runtime: {sequential_time} time units")
    print(f"  Actual runtime: {actual_runtime:.2f} time units")
    print(f"  Speedup: {efficiency:.2f}x")
    print(f"  Parallelization efficiency: {(efficiency * 100):.1f}%")
    
    # Group by start time to show parallel execution
    schedule_by_time = {}
    for task_id, start_time, embedding, adjusted_duration in schedule:
        if start_time not in schedule_by_time:
            schedule_by_time[start_time] = []
        schedule_by_time[start_time].append((task_id, embedding, adjusted_duration))
    
    max_parallel = max(len(tasks) for tasks in schedule_by_time.values()) if schedule_by_time else 1
    print(f"\nParallel execution details:")
    print(f"  Maximum parallel tasks: {max_parallel}")
    
    for start_time in sorted(schedule_by_time.keys()):
        parallel_tasks = schedule_by_time[start_time]
        print(f"  Time {start_time:.1f}: {len(parallel_tasks)} task(s) running")
        
        used_qubits = set()
        for task_id, embedding, adjusted_duration in parallel_tasks:
            physical_qubits = sorted(set(embedding.values()))
            used_qubits.update(physical_qubits)
            base_duration = next(t.d for t in tasks if t.task_id == task_id)
            swap_overhead = adjusted_duration - base_duration

            print(f"    Task {task_id}: qubits={physical_qubits}, duration={adjusted_duration:.2f}, original={base_duration:.2f}")

    print()
    
    total_runtime = scheduler.get_total_runtime(schedule)
    print(f"Total runtime: {total_runtime:.2f} time units")
    
    # Additional analysis
    total_base_time = sum(t.d for t in tasks)
    total_adjusted_time = sum(t.adjusted_duration for t in tasks if hasattr(t, 'adjusted_duration'))
    efficiency = total_base_time / total_adjusted_time if total_adjusted_time > 0 else 0
    
    print(f"Total base time: {total_base_time}")
    print(f"Total adjusted time: {total_adjusted_time:.2f}")
    print(f"SWAP overhead ratio: {((total_adjusted_time - total_base_time) / total_base_time * 100):.1f}%")


if __name__ == "__main__":
    main()