"""MCP Server for HERMES-Lite - exposes search as a tool for Claude Code."""
import os
import logging
import asyncio

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .database import Database
from .embed import EmbeddingClient
from .search import SearchEngine, Reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.environ.get('HERMES_CONFIG', '/app/config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# Initialize components
db = Database(config['database']['path'])
embed_client = EmbeddingClient(
    ollama_host=config['ollama']['host'],
    model=config['ollama']['embed_model'],
    batch_size=config['ollama']['embed_batch_size']
)

reranker = None
if config['reranker']['enabled']:
    reranker = Reranker(
        model_name=config['reranker']['model'],
        device=config['reranker']['device'],
        use_fp16=config['reranker']['use_fp16']
    )

search_engine = SearchEngine(
    database=db,
    embed_client=embed_client,
    reranker=reranker,
    semantic_weight=config['search']['semantic_weight'],
    keyword_weight=config['search']['keyword_weight'],
    rrf_k=config['search']['rrf_k']
)

# Create MCP server
server = Server("hermes-lite")


@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="hermes_search",
            description="Search the HERMES knowledge base for relevant documents and passages. Use this to find information from ingested papers, specs, and documentation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Use reranker for better relevance (default: true)",
                        "default": True
                    },
                    "doc_filter": {
                        "type": "string",
                        "description": "Optional filename pattern to filter results"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    if name != "hermes_search":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments.get("query")
    top_k = arguments.get("top_k", 10)
    rerank = arguments.get("rerank", True)
    doc_filter = arguments.get("doc_filter")

    if not query:
        raise ValueError("Query is required")

    logger.info(f"MCP search: {query[:50]}...")

    # Run search in thread pool since it's synchronous
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: search_engine.search(query, top_k, rerank, doc_filter)
    )

    # Format results for Claude
    if not results:
        return [TextContent(type="text", text="No results found.")]

    output_parts = []
    for i, r in enumerate(results, 1):
        output_parts.append(f"[{i}] {r['filename']}\n{r['content']}")

    return [TextContent(type="text", text="\n\n---\n\n".join(output_parts))]


async def main():
    """Run the MCP server."""
    logger.info("Starting HERMES-Lite MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
