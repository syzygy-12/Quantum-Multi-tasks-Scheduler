import networkx as nx
from typing import List, Tuple, Set, Dict, Optional
import itertools
import math

# ========== 原有调度与任务定义（保持不变） ==========

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
    
    def is_valid_embedding(self, task: Task, embedding: Dict[int, int]) -> bool:
        """Check if an embedding maintains connectivity."""
        # Check if all task topology edges are properly connected
        for (u, v) in task.task_topology.edges():
            physical_u = embedding[u]
            physical_v = embedding[v]
            
            # Check if physical qubits are connected in topology
            if not nx.has_path(self.topology, physical_u, physical_v):
                return False
        
        return True
    
    def find_connected_subgraphs(self, size: int, available_qubits: Set[int]) -> List[Set[int]]:
        """Find all connected subgraphs of given size using efficient BFS."""
        subgraphs = []
        available_list = list(available_qubits)
        
        # Use BFS to find connected components
        for start_node in available_list:
            visited = set([start_node])
            queue = [start_node]
            
            # BFS to find all reachable nodes within size limit
            while queue:
                node = queue.pop(0)
                if len(visited) == size:
                    subgraphs.append(visited.copy())
                    break
                
                for neighbor in self.topology.neighbors(node):
                    if neighbor in available_qubits and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        if len(visited) == size:
                            subgraphs.append(visited.copy())
                            break
        
        # Remove duplicates and filter by exact size
        unique_subgraphs = []
        seen = set()
        for subgraph in subgraphs:
            frozen = frozenset(subgraph)
            if len(subgraph) == size and frozen not in seen:
                seen.add(frozen)
                unique_subgraphs.append(subgraph)
        
        return unique_subgraphs
    
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

def create_custom_topology():
    G = nx.Graph()
    
    # 添加所有节点 (0-52)
    G.add_nodes_from(range(53))
    
    # 根据图片定义连接关系
    edges = [
        # 第一行: 0-1-2-3-4
        (0, 1), (1, 2), (2, 3), (3, 4),
        # 垂直连接
        (0, 5), (4, 6),
        
        # 第二行主体: 7-8-9-10-11-12-13-14-15
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
        # 垂直连接
        (5, 9), (6, 13), (7, 16), (11, 17), (15, 18),
        
        # 第三行: 19-20-21-22-23-24-25-26-27
        (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
        # 垂直连接
        (16, 19), (17, 23), (18, 27), (21, 28), (25, 29),
        
        # 第四行: 30-31-32-33-34-35-36-37-38
        (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
        # 垂直连接
        (28, 32), (29, 36), (34, 40), (38, 41), (30, 39),
        
        # 第五行: 42-43-44-45-46-47-48-49-50
        (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),
        # 垂直连接
        (39, 42), (40, 46), (41, 50), (44, 51), (48, 52),
    ]
    
    G.add_edges_from(edges)
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
    # topology = create_grid_topology(6, 6)
    topology=create_custom_topology()
    scheduler = QuantumScheduler(topology, swap_cost=1.0)
    
    # Define complex and intense quantum tasks
    # tasks = [
    #     Task(21, 2, 3, create_task_topology("line", 2)),
    #     Task(22, 3, 5, create_task_topology("line", 3)),
    #     Task(23, 4, 6, create_task_topology("star", 4)),
    #     Task(24, 5, 10, create_task_topology("ring", 5)),
    #     Task(25, 6, 12, create_task_topology("tree", 6)),
    #     Task(26, 7, 14, create_task_topology("grid", 7)),
    #     Task(27, 8, 16, create_task_topology("line", 8)),
    #     Task(28, 9, 18, create_task_topology("star", 9)),
    #     Task(29, 10, 20, create_task_topology("ring", 10)),
    #     Task(30, 11, 22, create_task_topology("tree", 11)),

    #     Task(31, 12, 25, create_task_topology("grid", 12)),
    #     Task(32, 13, 27, create_task_topology("line", 13)),
    #     Task(33, 14, 28, create_task_topology("star", 14)),
    #     Task(34, 15, 30, create_task_topology("ring", 15)),
    #     Task(35, 16, 35, create_task_topology("tree", 16)),
    #     Task(36, 17, 36, create_task_topology("grid", 17)),
    #     Task(37, 18, 38, create_task_topology("line", 18)),
    #     Task(38, 19, 40, create_task_topology("star", 19)),
    #     Task(39, 20, 42, create_task_topology("ring", 20)),
    #     Task(40, 21, 45, create_task_topology("tree", 21)),

    #     Task(41, 22, 48, create_task_topology("grid", 22)),
    #     Task(42, 23, 50, create_task_topology("line", 23)),
    #     Task(43, 24, 52, create_task_topology("star", 24)),
    #     Task(44, 25, 55, create_task_topology("ring", 25)),
    #     Task(45, 26, 58, create_task_topology("tree", 26)),
    #     Task(46, 27, 60, create_task_topology("grid", 27)),
    #     Task(47, 28, 62, create_task_topology("line", 28)),
    #     Task(48, 29, 65, create_task_topology("star", 29)),
    #     Task(49, 30, 68, create_task_topology("ring", 30)),
    #     Task(50, 31, 70, create_task_topology("tree", 31)),

    #     Task(51, 32, 72, create_task_topology("grid", 32)),
    #     Task(52, 33, 75, create_task_topology("line", 33)),
    #     Task(53, 34, 78, create_task_topology("star", 34)),
    #     Task(54, 35, 80, create_task_topology("ring", 35)),
    #     Task(55, 36, 82, create_task_topology("tree", 36)),
    #     Task(56, 15, 28, create_task_topology("grid", 15)),   # 中等复杂
    #     Task(57, 18, 40, create_task_topology("star", 18)),  # 大型星形
    #     Task(58, 20, 44, create_task_topology("ring", 20)),  # 大型环
    #     Task(59, 25, 55, create_task_topology("tree", 25)),  # 超复杂树
    #     Task(60, 30, 65, create_task_topology("grid", 30)),  # 超复杂格子
    # ]

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
    # 这两行只是把结果“导出”到全局，完全不改变调度计算
    # 在 main() 计算完 schedule 后的两行下面，再加一行
    globals()['__SCHEDULE__'] = schedule
    globals()['__TOPOLOGY__'] = topology
    globals()['__TASKS__'] = tasks   # ← 新增


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
            
            print(f"    Task {task_id}: qubits={physical_qubits}, duration={adjusted_duration:.2f}")
    
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

    # ========== 追加：每个调度时刻绘图，显示当时“所有正在运行”的任务 ==========
    # （仅读取上面的 schedule 结果，不更改任何调度计算过程或数据）
    import os
    import matplotlib.pyplot as plt

        
    def get_custom_node_positions():
        """定义节点在图中的位置坐标，模拟实际拓扑布局"""
        positions = {}
    
    # 第一行 (y=4)
        for i, node in enumerate([0, 1, 2, 3, 4]):
            positions[node] = (i+2.5, 4)
    
    # 特殊连接点
        positions[5] = (2.5, 3)
        positions[6] = (6.5, 3)
    
    # 第二行 (y=2)
        for i, node in enumerate([7, 8, 9, 10, 11, 12, 13, 14, 15]):
            positions[node] = (i + 0.5, 2)
    
    # 垂直分支
        positions[16] = (0.5, 1)
        positions[17] = (4.5, 1)
        positions[18] = (8.5, 1)
    
    # 第三行 (y=0)
        for i, node in enumerate([19, 20, 21, 22, 23, 24, 25, 26, 27]):
            positions[node] = (i + 0.5, 0)
    
    # 垂直分支
        positions[28] = (2.5, -1)
        positions[29] = (6.5, -1)
    
    # 第四行 (y=-2)
        for i, node in enumerate([30, 31, 32, 33, 34, 35, 36, 37, 38]):
            positions[node] = (i + 0.5, -2)
    
    # 垂直分支
        positions[39] = (0.5, -3)
        positions[40] = (4.5, -3)
        positions[41] = (8.5, -3)
    
    # 第五行 (y=-4)
        for i, node in enumerate([42, 43, 44, 45, 46, 47, 48, 49, 50]):
            positions[node] = (i + 0.5, -4)
    
    # 最底层分支
            positions[51] = (2.5, -5)
            positions[52] = (6.5, -5)
    
        return positions

    def _node_to_rc(node: int, cols: int) -> Tuple[int, int]:
        return node // cols, node % cols

    def _draw_chip_frame(topology: nx.Graph,
                         time_t: float,
                         active_tasks: List[Tuple[int, Dict[int, int]]],
                         rows: int, cols: int,
                         out_path: str):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"Active tasks on chip at time {time_t} (ALL running tasks)")
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True, which="both")
        ax.invert_yaxis()

        # 画物理边
        for (n1, n2) in topology.edges():
            r1, c1 = _node_to_rc(n1, cols); r2, c2 = _node_to_rc(n2, cols)
            ax.plot([c1, c2], [r1, r2], linewidth=1, alpha=0.3)

        # 画任务占用
        handles, labels = [], []
        for tid, emb in sorted(active_tasks, key=lambda x: x[0]):
            xs, ys = [], []
            for q in set(emb.values()):
                r, c = _node_to_rc(q, cols)
                xs.append(c); ys.append(r)
            sc = ax.scatter(xs, ys, s=300, alpha=0.85, label=f"Task {tid}")
            for x, y in zip(xs, ys):
                ax.text(x, y, str(tid), ha="center", va="center", fontsize=9)
            handles.append(sc); labels.append(f"Task {tid}")

        if handles:
            ax.legend(handles=handles, labels=labels, loc="upper right", bbox_to_anchor=(1.25, 0.2))

        ax.set_xlabel("Column"); ax.set_ylabel("Row")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    
    def _draw_chip_frame_custom(topology: nx.Graph,
                           time_t: float,
                           active_tasks: List[Tuple[int, Dict[int, int]]],
                           out_path: str):
        """使用自定义拓扑绘制芯片状态"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(f"Active tasks on chip at time {time_t} (Custom Topology)")
    
    # 获取节点位置
        pos = get_custom_node_positions()
    
    # 设置坐标范围
        x_coords = [pos[node][0] for node in pos]
        y_coords = [pos[node][1] for node in pos]
        x_margin = (max(x_coords) - min(x_coords)) * 0.1
        y_margin = (max(y_coords) - min(y_coords)) * 0.1
    
        ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
        ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # 画物理连接边
        for (n1, n2) in topology.edges():
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.4)
    
    # 画所有节点（空心圆）
        for node in topology.nodes():
            x, y = pos[node]
            ax.plot(x, y, 'o', markersize=12, markerfacecolor='lightblue', 
               markeredgecolor='black', markeredgewidth=1)
            # ax.text(x, y, str(node), ha='center', va='center', fontsize=8, fontweight='bold')
    
    # 画任务占用（彩色填充）
        handles, labels = [], []
        colors = plt.cm.Set3(range(len(active_tasks)))
    
        for idx, (tid, emb) in enumerate(sorted(active_tasks, key=lambda x: x[0])):
            occupied_nodes = list(set(emb.values()))
            xs, ys = [], []
            for node in occupied_nodes:
                if node in pos:
                    x, y = pos[node]
                    xs.append(x)
                    ys.append(y)
        
            if xs:  # 如果有有效的节点位置
                color = colors[idx % len(colors)]
                sc = ax.scatter(xs, ys, s=300, alpha=0.8, c=[color], 
                           label=f"Task {tid}", edgecolors='black', linewidths=2)
            
            # 在节点上标记任务ID
                for x, y in zip(xs, ys):
                    ax.text(x, y, str(tid), ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='black')
            
                handles.append(sc)
                labels.append(f"Task {tid}")
    
        if handles:
            ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.02, 1))
    
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    # 事件时间：所有开始时间
    event_times = sorted({s for (_tid, s, _emb, _dur) in schedule})

    # tid -> (start, end, emb)
    task_time = {}
    for tid, s, emb, dur in schedule:
        task_time[tid] = (s, s + dur, emb)

    # 这里与上面 create_grid_topology(6,6) 保持一致
    rows, cols = 6, 6
    out_dir = "chip_schedules_full"
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    for t in event_times:
        active = [(tid, emb) for tid, (s, e, emb) in task_time.items() if s <= t < e]
        out_path = f"{out_dir}/active_at_{t}.png"
        # _draw_chip_frame(topology, t, active, rows, cols, out_path)
        _draw_chip_frame_custom(topology, t, active, out_path)  # 使用自定义布局
        frames.append(out_path)

    with open(f"{out_dir}/frames_index.txt", "w", encoding="utf-8") as f:
        for p in frames:
            f.write(p + "\\n")

    with open(f"{out_dir}/index.html", "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\\n")
        f.write("<title>Chip Schedules (All Active Tasks)</title>\\n")
        f.write("<h1>Chip Schedules (All Active Tasks)</h1>\\n")
        for p in frames:
            name = p.split("/")[-1]
            f.write(f"<div style='margin:12px 0'><h3>{name}</h3><img src='{name}' style='max-width:100%;height:auto;border:1px solid #ddd'/></div>\\n")

    print(f"[viz] 已生成 {len(frames)} 张图，目录：{out_dir}")
    print(f"[viz] 图集：{out_dir}/index.html")



if __name__ == "__main__":
    main()

# ========================= 追加：生成 GIF 并高亮“正在操作”的比特 =========================
# 说明：
# - 不修改上面的任何调度/嵌入/计算逻辑，只复用已得到的 schedule/topology。
# - “正在操作”的比特：在该时刻 t 新开始的任务（start==t）的所有嵌入物理比特，用红色空心圆标记。
# - 输出到与图片相同的目录 out_dir（默认 chip_schedules_full），文件名 active_tasks.gif。

import os as _gif_os
from pathlib import Path as _gif_Path
from typing import Dict as _gif_Dict, List as _gif_List, Tuple as _gif_Tuple
import matplotlib.pyplot as _gif_plt
from PIL import Image as _gif_Image

def _gif_node_to_rc(_n: int, _cols: int):
    return _n // _cols, _n % _cols

def _gif_draw_frame(_topology, _time_t: float,
                    _active: _gif_List[_gif_Tuple[int, dict]],
                    _starting: _gif_List[_gif_Tuple[int, dict]],
                    _rows: int, _cols: int,
                    _out_path: str):
    fig, ax = _gif_plt.subplots(figsize=(7, 7))
    ax.set_title(f"Active tasks on chip at time { _time_t } (ALL running tasks)")
    ax.set_xlim(-0.5, _cols - 0.5); ax.set_ylim(-0.5, _rows - 0.5)
    ax.set_xticks(range(_cols)); ax.set_yticks(range(_rows))
    ax.grid(True, which="both"); ax.invert_yaxis()

    # 画物理连边
    for (n1, n2) in _topology.edges():
        r1, c1 = _gif_node_to_rc(n1, _cols); r2, c2 = _gif_node_to_rc(n2, _cols)
        ax.plot([c1, c2], [r1, r2], linewidth=1, alpha=0.3)

    # 画所有正在运行任务（与原占用完全一致）
    handles, labels = [], []
    for tid, emb in sorted(_active, key=lambda x: x[0]):
        xs, ys = [], []
        for q in set(emb.values()):
            r, c = _gif_node_to_rc(q, _cols)
            xs.append(c); ys.append(r)
        sc = ax.scatter(xs, ys, s=300, alpha=0.85, label=f"Task {tid}")
        for x, y in zip(xs, ys):
            ax.text(x, y, str(tid), ha="center", va="center", fontsize=9)
        handles.append(sc); labels.append(f"Task {tid}")
    if handles:
        ax.legend(handles=handles, labels=labels, loc="upper right", bbox_to_anchor=(1.25, 0.2))

    # 用红色空心圆高亮：本时刻新开始的任务的比特（正在进行门操作）
    hi_x, hi_y = [], []
    for tid, emb in _starting:
        for q in set(emb.values()):
            r, c = _gif_node_to_rc(q, _cols)
            hi_x.append(c); hi_y.append(r)
    if hi_x:
        ax.scatter(hi_x, hi_y, s=700, facecolors='none', edgecolors='red', linewidths=2)

    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    fig.savefig(_out_path, bbox_inches="tight", dpi=150)
    _gif_plt.close(fig)

def get_custom_node_positions():
        """定义节点在图中的位置坐标，模拟实际拓扑布局"""
        positions = {}
    
    # 第一行 (y=4)
        for i, node in enumerate([0, 1, 2, 3, 4]):
            positions[node] = (i * 2 + 2, 4)
    
    # 特殊连接点
        positions[5] = (0, 3)
        positions[6] = (8, 3)
    
    # 第二行 (y=2)
        for i, node in enumerate([7, 8, 9, 10, 11, 12, 13, 14, 15]):
            positions[node] = (i + 0.5, 2)
    
    # 垂直分支
        positions[16] = (0.5, 1)
        positions[17] = (4.5, 1)
        positions[18] = (8.5, 1)
    
    # 第三行 (y=0)
        for i, node in enumerate([19, 20, 21, 22, 23, 24, 25, 26, 27]):
            positions[node] = (i + 0.5, 0)
    
    # 垂直分支
        positions[28] = (2.5, -1)
        positions[29] = (6.5, -1)
    
    # 第四行 (y=-2)
        for i, node in enumerate([30, 31, 32, 33, 34, 35, 36, 37, 38]):
            positions[node] = (i + 0.5, -2)
    
    # 垂直分支
        positions[39] = (0.5, -3)
        positions[40] = (4.5, -3)
        positions[41] = (8.5, -3)
    
    # 第五行 (y=-4)
        for i, node in enumerate([42, 43, 44, 45, 46, 47, 48, 49, 50]):
            positions[node] = (i + 0.5, -4)
    
    # 最底层分支
            positions[51] = (4.5, -5)
            positions[52] = (6.5, -5)
    
        return positions

def _gif_draw_frame_custom(_topology, _time_t: float,
                          _active: List[Tuple[int, dict]],
                          _starting: List[Tuple[int, dict]],
                          _out_path: str):
    """为GIF生成自定义拓扑帧，高亮正在执行的门操作"""
    fig, ax = _gif_plt.subplots(figsize=(12, 8))
    ax.set_title(f"Active tasks on chip at time {_time_t} (Gate Operations Highlighted)")
    
    # 获取节点位置
    pos = get_custom_node_positions()
    

def generate_animation_with_gate_marks(_schedule, _topology, out_dir: str = "chip_schedules_full",
                                       rows: int = 6, cols: int = 6,
                                       duration_ms: int = 800, gif_name: str = "active_tasks.gif"):
    # 事件时间：所有开始时间
    event_times = sorted({s for (_tid, s, _emb, _dur) in _schedule})
    # tid -> (start, end, emb)
    task_time = {tid: (s, s + dur, emb) for (tid, s, emb, dur) in _schedule}

    _gif_os.makedirs(out_dir, exist_ok=True)
    frame_paths = []

    for t in event_times:
        active   = [(tid, emb) for tid, (s, e, emb) in task_time.items() if s <= t < e]
        starting = [(tid, emb) for tid, (s, e, emb) in task_time.items() if s == t]  # 正在“开始执行门”的比特
        out_path = str(_gif_Path(out_dir) / f"active_at_{t}_op.png")
        # _gif_draw_frame(_topology, t, active, starting, rows, cols, out_path)
        _gif_draw_frame_custom(_topology, t, active, starting, out_path)
        frame_paths.append(out_path)

    # 组装 GIF（与图片在同一目录）
    frames = [_gif_Image.open(p).convert("RGB") for p in frame_paths]
    gif_path = str(_gif_Path(out_dir) / gif_name)
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
        print(f"[gif] 已生成动画：{gif_path}  （共 {len(frames)} 帧）")
    else:
        print("[gif] 未生成帧，检查 schedule 是否为空。")

# —— 调用：在主程序末尾（schedule 计算完成之后）追加执行 —— 
# 第一处 GIF（带 gate marks）
try:
    _S = globals().get('__SCHEDULE__')
    _T = globals().get('__TOPOLOGY__')
    if _S is None or _T is None:
        raise RuntimeError("未导出 __SCHEDULE__/__TOPOLOGY__。请在计算 schedule 后添加两行 globals()[...] = ...")

    generate_animation_with_gate_marks(_S, _T,  # ← 用 _S、_T 而不是 schedule, topology
                                       out_dir="chip_schedules_full",
                                       rows=6, cols=6,
                                       duration_ms=800, gif_name="active_tasks.gif")
except Exception as _gif_e:
    print("[gif] 生成动画失败：", _gif_e)
# ========================= 追加代码结束 =========================
# ========================= 追加：逐 runtime 帧 + GIF，高亮正在执行门的比特 =========================
# 说明：
# - 仅复用已得到的 schedule / tasks / topology；不改动调度计算。
# - 每个整数时间步（runtime=1 的粒度）生成一帧，占用与原图一致；
#   对每个活跃任务，按其逻辑边 round-robin 选一条边作为当前门，并用红色空心圆标出该边两端物理比特。
# - 输出目录沿用 chip_schedules_full，并生成 runtime 级别帧与 GIF：runtime_active_*.png / runtime_active.gif

# import math as _rt_math
# import os as _rt_os
# from pathlib import Path as _rt_Path
# import matplotlib.pyplot as _rt_plt
# from PIL import Image as _rt_Image

# def _rt_node_to_rc(_n: int, _cols: int):
#     return _n // _cols, _n % _cols

# def _rt_draw_frame(_topology, _time_t: int,
#                    _active,                # List[(tid, embedding, task_obj)]
#                    _operating_phys_nodes,  # Set[int]
#                    _rows: int, _cols: int,
#                    _out_path: str):
#     fig, ax = _rt_plt.subplots(figsize=(7, 7))
#     ax.set_title(f"Active tasks at runtime { _time_t } (ALL running; red ring = gate qubits)")
#     ax.set_xlim(-0.5, _cols - 0.5); ax.set_ylim(-0.5, _rows - 0.5)
#     ax.set_xticks(range(_cols)); ax.set_yticks(range(_rows))
#     ax.grid(True, which="both"); ax.invert_yaxis()

#     # 物理连边
#     for (n1, n2) in _topology.edges():
#         r1,c1 = _rt_node_to_rc(n1,_cols); r2,c2 = _rt_node_to_rc(n2,_cols)
#         ax.plot([c1,c2],[r1,r2], linewidth=1, alpha=0.3)

#     # 画所有活跃任务占用（与原占用一致）
#     handles, labels = [], []
#     for tid, emb, _task in sorted(_active, key=lambda x: x[0]):
#         xs, ys = [], []
#         for q in set(emb.values()):
#             r, c = _rt_node_to_rc(q, _cols)
#             xs.append(c); ys.append(r)
#         sc = ax.scatter(xs, ys, s=300, alpha=0.85, label=f"Task {tid}")
#         for x, y in zip(xs, ys):
#             ax.text(x, y, str(tid), ha="center", va="center", fontsize=9)
#         handles.append(sc); labels.append(f"Task {tid}")
#     if handles:
#         ax.legend(handles=handles, labels=labels, loc="upper right", bbox_to_anchor=(1.25, 0.2))

#     # 高亮：正在执行门的比特（红色空心圆）
#     if _operating_phys_nodes:
#         hi_x, hi_y = [], []
#         for q in _operating_phys_nodes:
#             r, c = _rt_node_to_rc(q, _cols)
#             hi_x.append(c); hi_y.append(r)
#         ax.scatter(hi_x, hi_y, s=700, facecolors='none', edgecolors='red', linewidths=2)

#     ax.set_xlabel("Column"); ax.set_ylabel("Row")
#     fig.savefig(_out_path, bbox_inches="tight", dpi=150)
#     _rt_plt.close(fig)

# def generate_per_runtime_gif(_schedule, _tasks, _topology,
#                              out_dir: str = "chip_schedules_full",
#                              rows: int = 6, cols: int = 6,
#                              step: int = 1, gif_name: str = "runtime_active.gif",
#                              duration_ms: int = 80):
#     """
#     每个整数 runtime 生成一帧；按任务逻辑边 round-robin 选择当前门并高亮对应物理比特。
#     step=1 表示每个时间单位一帧；可按需调大减小帧数。
#     """
#     # 总时长（向上取整）
#     total_runtime = 0
#     for tid, s, emb, dur in _schedule:
#         total_runtime = max(total_runtime, int(_rt_math.ceil(s + dur)))
#     if total_runtime <= 0:
#         print("[runtime-gif] 空调度，跳过。")
#         return

#     # 辅助表：tid -> (start, end, embedding, task_obj)
#     # 建 task_id -> Task 的字典，便于拿到逻辑边
#     _task_by_id = {t.task_id: t for t in _tasks}
#     _info = {}
#     for tid, s, emb, dur in _schedule:
#         _info[tid] = (int(_rt_math.floor(s)), int(_rt_math.ceil(s + dur)), emb, _task_by_id.get(tid))

#     _rt_os.makedirs(out_dir, exist_ok=True)
#     frame_paths = []

#     for t in range(0, total_runtime, step):
#         # 本时刻活跃任务
#         active = []
#         for tid, (s, e, emb, t_obj) in _info.items():
#             if s <= t < e:
#                 active.append((tid, emb, t_obj))

#         # 计算“正在执行门”的物理比特集合
#         operating = set()
#         for tid, emb, t_obj in active:
#             if t_obj is None:
#                 # 没有任务对象就无法取逻辑边——保守：不标
#                 continue
#             edges = list(t_obj.task_topology.edges())
#             if not edges:
#                 # 边为空（比如 size<2），这里不标或标一个单比特也可；按需你可改为标所有占用比特
#                 continue
#             # 在边列表上 round-robin：用 (t - start) % len(edges)
#             s, e, _, _ = _info[tid]
#             idx = (t - s) % len(edges)
#             u, v = edges[idx]
#             pu, pv = emb.get(u), emb.get(v)
#             if pu is not None: operating.add(pu)
#             if pv is not None: operating.add(pv)

#         # 画帧
#         out_path = str(_rt_Path(out_dir) / f"runtime_active_{t}.png")
#         _rt_draw_frame(_topology, t, active, operating, rows, cols, out_path)
#         frame_paths.append(out_path)

#     # 合成 GIF
#     frames = [_rt_Image.open(p).convert("RGB") for p in frame_paths[:40]]
#     if frames:
#         gif_path = str(_rt_Path(out_dir) / gif_name)
#         frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
#         print(f"[runtime-gif] 已生成 GIF：{gif_path}（{len(frames)} 帧，step={step}）")
#     else:
#         print("[runtime-gif] 未生成帧。")

# # —— 调用：放在主程序 schedule 与 tasks 计算完成之后（例如你现有绘图/统计之后）——
# # 第二处 runtime 级别的 GIF
# try:
#     _S  = globals().get('__SCHEDULE__')
#     _T  = globals().get('__TOPOLOGY__')
#     _TS = globals().get('__TASKS__')          # ← 新增：取出 tasks
#     if _S is None or _T is None or _TS is None:
#         raise RuntimeError("未导出 __SCHEDULE__/__TOPOLOGY__/__TASKS__。")

#     generate_per_runtime_gif(_S, _TS, _T,     # ← 用 _S、_TS、_T
#                              out_dir="chip_schedules_full",
#                              rows=6, cols=6,
#                              step=1,
#                              gif_name="runtime_active.gif",
#                              duration_ms=60)
# except Exception as _e:
#     print("[runtime-gif] 生成失败：", _e)

# # ========================= 追加代码结束 =========================
# # 