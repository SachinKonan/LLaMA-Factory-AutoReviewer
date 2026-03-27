"""Stdio MCP server that wraps the retrieval server with submission_id -> year filtering."""

import json
import os
from pathlib import Path

import httpx
from datasets import load_from_disk
from fastmcp import FastMCP

METADATA_PATH = Path(__file__).resolve().parents[2] / "data" / "massive_metadata_v7_5"
RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://127.0.0.1:8000/retrieve")

# Load submission_id -> year mapping once at import time
_ds = load_from_disk(str(METADATA_PATH), keep_in_memory=False)
ID_TO_YEAR: dict[str, int] = {
    row["submission_id"]: row["year"] for row in _ds.select_columns(["submission_id", "year"])
}
del _ds

mcp = FastMCP("arxiv_search")


@mcp.tool()
async def search_arxiv(submission_id: str, query: str, topk: int = 5) -> str:
    """Search arxiv papers relevant to a query, filtered to papers available before the given submission's review period.

    Args:
        submission_id: The ICLR submission ID of the paper being reviewed.
                       Used to determine the date cutoff so only prior work is returned.
        query: Search query (e.g. paper title, method name, topic).
        topk: Number of results to return (default 5).

    Returns:
        Formatted search results with titles and abstracts of matching arxiv papers.
    """
    year = ID_TO_YEAR.get(submission_id)
    if year is None:
        return f"Error: submission_id '{submission_id}' not found in metadata."

    upper_bound = f"{year - 1}-12-15"

    payload = {
        "query": query,
        "topk": topk,
        "upper_bound_datetime": upper_bound,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(RETRIEVAL_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("result", [[]])
    if not results or not results[0]:
        return f"No results found for query '{query}' (cutoff: {upper_bound})."

    lines = [f"Search results for '{query}' (papers before {upper_bound}):\n"]
    for i, doc in enumerate(results[0], 1):
        if isinstance(doc, dict):
            contents = doc.get("contents", doc.get("document", {}).get("contents", ""))
            title = doc.get("title", doc.get("document", {}).get("title", "Unknown"))
        else:
            contents = str(doc)
            title = "Unknown"
        lines.append(f"--- Result {i} ---")
        lines.append(f"Title: {title}")
        lines.append(contents)
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
