#!/usr/bin/env python3
"""
Ultra-simple MCP topology creation service.

Create quantum computer topologies from nodes and edges.
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import networkx as nx
import json
from typing import List
from pathlib import Path
import logging

# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

mcp = FastMCP("create_topology")

@mcp.tool()
def create_topology(nodes: List[int], edges: List[List[int]], name: str) -> TextContent:
    """Create topology from nodes and edges.
    
    Args:
        nodes: List of node IDs (e.g., [0, 1, 2, 3])
        edges: List of edges as [u, v] pairs (e.g., [[0,1], [1,2], [2,3]])
        name: Name for this topology
    
    Returns:
        JSON string with topology information
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

def main():
    logger.info('Starting your-new-server')
    mcp.run('stdio')

if __name__ == "__main__":
    main()