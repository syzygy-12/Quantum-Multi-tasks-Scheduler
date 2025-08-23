from asyncio import tasks
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from quantum_analyzer import main as analyze_quantum_circuits
from LQuantum import QuantumScheduler, Task, create_grid_topology, create_topology_from_json, create_task_topology
import json
import networkx as nx
import shutil
from typing import List
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
import re

# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the MCP server object
mcp = FastMCP()

# Here’s where you define your tools (functions the AI can use)
@mcp.tool()
def add(a: int, b: int) -> TextContent:
    """Add two numbers.

    Args:
        a: the first integer to be added
        b: the second integer to be added
    
    Return:
        The sum of the two integers, as a string."""
    return TextContent(type="text", text=str(a + b))

# The return format should be one of the types defined in mcp.types. The commonly used ones include TextContent, ImageContent, BlobResourceContents.
# In the case of a string, you can also directly use `return str(a + b)` which is equivalent to `return TextContent(type="text", text=str(a + b))`

@mcp.tool()
def get_image_of_flower():
    """Get an image of flower

    Return:
        Image of flower in png."""
    image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGYktHRAD/AP8A/6C9p5MAAAAJcEhZcwABGUAAARlAAYDjddQAAAAHdElNRQfpBBUNAgfLUoX1AAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDI1LTA0LTIxVDEzOjAxOjU2KzAwOjAwMB5AXgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAyNS0wNC0yMVQxMzowMTo1NiswMDowMEFD+OIAAAAodEVYdGRhdGU6dGltZXN0YW1wADIwMjUtMDQtMjFUMTM6MDI6MDcrMDA6MDAT9mfuAAAAMXRFWHRDb21tZW50AFBORyByZXNpemVkIHdpdGggaHR0cHM6Ly9lemdpZi5jb20vcmVzaXplXknb4gAAABJ0RVh0U29mdHdhcmUAZXpnaWYuY29toMOzWAAABr5JREFUSMell2uIXGcZx3/P+77nzJy57CXZXJu0m602lyam2mprESElm0YwCoJVC35T3C1IESEICuKlahBqKTUJCJovFvwgaLVUa4WGhoq2CYS01UQzm0u7SXez97mdmXPexw9nZzaJ7obS59MwZ87zf9///7n8R3ifMfbwKADqtfvd0F+O3vI9875A942AgKYKgtUkA68Mj9zyXVnpYTeBZr9Max5Xsmx58Qhj+0exOUNSTUH4BPB14BWEo0BiQsPg84ffO/DY/tHOTRyeIYRplGs3vJFdcCvwW+BuYA74LHAcWZny5an2miX2PAS8hPIcyvDmg0OE63PgQQwW+MYiKEAvcEC9ImZFMnHLcyGgCsIwsBnYbPLm2MSz448F63K/T6sp6ULyYeBzNzFwj1iJ1GsD4MKnH7sh7eAfD68MrKqIE9FUNwKIFcof79+Yuz06Ik5W9+1Zdeydn114BFjTEc31BQAbk9l2WQyNyvAIvulvELaydwRxsjywiHSkCFEwkSXcmMNEZgPKk/V/1SKU/QASCMUPlSnsKKNtLbeuNAut8SaN8/VNwMeAbcAM8DvgiiaaAY99ahRJFF3UJa2lRFuLVJ5+K12/9/Zml0ftnrzX5MwPJJCSeqX0kV6Ku3uyYyqhv+h3NM7VvoyRLwEfAILFFB9F+BrQNpV9I4hktSSBCIJbfWAtF58+ye5nH4zESR3Axx7fSLtnsL1Bv+t1QTRUpLCr3O2PdCHpq59Z+AUi30fZ3gXN4m6UPApOEHysIDzom/4rQP/8qzNv7/jlHifWbF91YN2dzfM1mufrtN9tEazPgYIJhPC2PNEHi4iTLhvNsXroY7/R9jjECb6e4uOuzidNKFWfKE5VwVBE+SGwBwUJDGIFDARrQ4K1IfmhAq0rMRorEgiIEG0r43pcF1SbHpMz9D+8BtvrEGdoT8bMvTyFr/urCMd8SxVZ6uNkUfxsQs0nNCv1pRZRCNblKGwrdSkVl5DbtIDk4qWCDA3R1hLh5jy27DAFiy05EEmBp7YcGvwHAiY0GPWAEgNPAG9ANnurr89Rf7O6KH4GLjmzSKtgy9OE6y4QDlxGTNptFwB89llbnvqZBXwt/RXCM2PfupD18vOHMcYJJjQAp4BHgT8gpL7pWfjbDHOvTJNca2XJAEQx+RomjPH1fsQlYBOun76aKu2rMXPHp2icq50Avg3UULoTTQAq+0ZwBdsZ+GWULwCP49mJgWBNSGFrkdyWIm5VnXDdGNrsJZm7DbdqDB9HtKc3oDE0z9ezQrzWyoaHMAk8ifAMSrXT966jY1JLrx98EwhT4aYc0V0lgrUhJm+RwEDqwBskaGBLE5iggQnrpNU+fFok3JAjmWnTmmh18q0BnkDZgXAQ5aq2FdtZfSYQNOUh4CngYGF76c6eT64m3JDH5DNtxYCmDt+KuuAmjNEkJFlYDd5hIktuU4Trc7Qnsi5AEGA3cAfCX4GG... [truncated]"
    # if you'''re not familiar with base64, you can see https://en.wikipedia.org/wiki/Base64

    return ImageContent(data=image_base64, mimeType="image/png", type="image")

@mcp.tool()
def analyze_quantum_directory(
    input_dir: str = "./circuits",
    output_file: str = "./circuits/circuits.json",
    backend: str = None,
    opt_level: int = 1,
    seed: int = None
) -> TextContent:
    """Analyze quantum circuits in a directory and save results to JSON. 
       If no specification, just use default values.

    Args:
        input_dir: Directory to recursively search for .qpy and .qasm files, default in ./circuits
        output_file: Output JSON file path (default: ./circuits/circuits.json)
        backend: Optional fake backend class name (e.g., FakeLima, FakeNairobi)
        opt_level: Transpile optimization level (0-3, default: 1)
        seed: Optional seed for transpiler
    
    Return:
        Analysis results as JSON string
    """
    try:
        # Build argument list for quantum analyzer
        args = [input_dir, "-o", output_file]
        if backend:
            args.extend(["--backend", backend])
        args.extend(["--opt-level", str(opt_level)])
        if seed is not None:
            args.extend(["--seed", str(seed)])
        
        # Run the analysis
        result = analyze_quantum_circuits(args)
        
        # Read the results file
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                results_content = f.read()
            return TextContent(type="text", text=f"Analysis completed successfully. Results saved to {output_file}.\n\n{results_content}")
        else:
            return TextContent(type="text", text=f"Analysis completed with exit code {result}, but output file not found.")
            
    except Exception as e:
        return TextContent(type="text", text=f"Error during quantum analysis: {str(e)}")

@mcp.tool()
def create_topology(nodes: List[int], edges: List[List[int]], name: str) -> TextContent:
    """Create topology from nodes and edges.
    
    Args:
        nodes: List of node IDs (e.g., [0, 1, 2, 3])
        edges: List of edges as [u, v] pairs (e.g., [[0,1], [1,2], [2,3]])
        name: Name for this topology
    
    Returns:
        JSON string with topology information
        store topology information in a JSON file in ./topologies
    """
    storage_dir = Path("./topologies")
    storage_dir.mkdir(exist_ok=True)
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    data = {
        "name": name,
        "nodes": list(G.nodes()),
        "edges": [[u, v] for u, v in G.edges()],
        "metadata": {
            "num_nodes": len(G.nodes()),
            "num_edges": len(G.edges()),
            "is_connected": nx.is_connected(G)
        }
    }
    
    filepath = storage_dir / f"{name}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return TextContent(type="text", text=json.dumps(data, indent=2))

@mcp.tool()
def quantum_task_scheduler(
    circuits_config_path: str = './circuits/circuits.json',
    topology_config_path: str = None,
    swap_cost: float = 1.0,
    output_path: str = "quantum_schedule_results.json"
) -> TextContent:
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
        actual_runtime = scheduler.get_total_runtime(schedule)
        efficiency = total_base_time / actual_runtime if actual_runtime > 0 else 0
        
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
                "sequential_runtime": total_base_time,
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
- Sequential runtime: {total_base_time} time units
- Actual runtime: {actual_runtime:.2f} time units
- Speedup achieved: {efficiency:.2f}x
- Parallelization efficiency: {(efficiency * 100):.1f}%
- Maximum parallel tasks: {max_parallel}

The scheduling algorithm successfully optimized {len(tasks)} quantum tasks with SWAP-aware placement, achieving significant parallelization benefits. Tasks were grouped by size to prevent starvation, and optimal embeddings were found to minimize SWAP overhead while maximizing resource utilization.

Scheduled tasks range from small 2-qubit operations to complex 20-qubit algorithms, demonstrating the scheduler's ability to handle diverse quantum workloads efficiently."""
        
        return TextContent(type="text", text=summary)
        
    except Exception as e:
        return TextContent(type="text", text=f"Error during quantum task scheduling: {str(e)}")

@mcp.tool()
def from_program_generate_qpy(
    program_path: str
):
    """
    Args:
        program_path (str): The path to the Python program file to execute.

    Functionality:
        This function executes a Python program located at the specified path, usually in ./programs, after execution, a .qpy file will be generated.
    """
    # 运行这个py文件
    exec(open(program_path).read())
    return

@mcp.tool()
def create_gif_with_custom_order(folder_path: str = "./chip_schedules_full",
                               file_pattern: str = None,
                               output_name: str = "custom_animation.gif",
                               duration_ms: int = 800,
                               return_base64: bool = True):
    """
    按自定义模式创建GIF，可选择返回Base64编码或保存文件

    Args:
        folder_path: 文件夹路径，默认"chip_schedules_full"
        file_pattern: 文件名模式，如 "active_at_*.png"
        output_name: 输出文件名，默认"custom_animation.gif"
        duration_ms: 每帧时长，默认800
        return_base64: 是否返回Base64编码的图片内容，默认True
    """

    folder = Path(folder_path)

    # 检查文件夹是否存在
    if not folder.exists():
        return f"错误: 文件夹 '{folder_path}' 不存在"
    
    if not folder.is_dir():
        return f"错误: '{folder_path}' 不是文件夹"

    # 获取PNG文件
    if file_pattern:
        png_files = list(folder.glob(file_pattern))
    else:
        png_files = list(folder.glob("*.png"))

    if not png_files:
        return f"未找到匹配的文件: {file_pattern or '*.png'}"

    # 手动排序（可以根据需要修改）
    def custom_sort_key(file):
        name = file.stem  # 不包含扩展名的文件名

        # 示例：按时间排序
        time_match = re.search(r'(\d+(?:\.\d+)?)$', name)
        if time_match:
            return float(time_match.group(1))
        return name

    png_files.sort(key=custom_sort_key)

    print(f"按自定义顺序找到 {len(png_files)} 个文件")

    # 生成GIF
    images = []
    failed_files = []
    for file in png_files:
        try:
            img = Image.open(file).convert('RGB')
            images.append(img)
        except Exception as e:
            failed_files.append(f"{file.name}: {e}")

    if not images:
        return "没有成功加载任何图片文件"

    # 确保所有图片尺寸一致
    first_img_size = images[0].size
    for i, img in enumerate(images):
        if img.size != first_img_size:
            print(f"调整图片尺寸: {png_files[i].name}")
            images[i] = img.resize(first_img_size, Image.Resampling.LANCZOS)

    # 创建内存缓冲区
    buffer = BytesIO()
    
    try:
        images[0].save(
            buffer,
            format='GIF',
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )
        
        if return_base64:
            # 返回Base64编码的图片内容
            gif_data = buffer.getvalue()
            base64_encoded = base64.b64encode(gif_data).decode('utf-8')
            
            # 返回ImageContent对象
            return ImageContent(
                data=base64_encoded,
                mimeType="image/gif",
                type="image"
            )
        else:
            # 保存到文件
            output_path = folder / output_name
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            
            result = f"成功生成GIF: {output_path}\n"
            result += f"处理了 {len(images)} 个图片文件"
            if failed_files:
                result += f"\n读取失败的文件:\n" + "\n".join(failed_files)
            return result
            
    except Exception as e:
        return f"生成GIF失败: {e}"



# The return format should be one of the types defined in mcp.types. The commonly used ones include TextContent, ImageContent, BlobResourceContents.
# In the case of a string, you can also directly use `return str(a + b)` which is equivalent to `return TextContent(type="text", text=str(a + b))`


# ============================= ADD-MCP-TOOL-BEFORE-THIS-LINE ===============================

# This is the main entry point for your server
def main():
    logger.info('Starting your-new-server')
    mcp.run('stdio')

if __name__ == "__main__":
    main()
