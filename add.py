from mcp.server.fastmcp import FastMCP
import base64
import logging
from mcp.types import TextContent, ImageContent
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP()

@mcp.tool()
async def add(a: int, b: int) -> str:
    """Add two numbers together.

    This tool takes two numbers as input and returns the result of adding them together.
    """
    logger.info('Adding numbers')
    return str(a + b)

def main():
    # Start server
    logger.info('Starting example-server')
    mcp.run('stdio')