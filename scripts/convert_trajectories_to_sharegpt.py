#!/usr/bin/env python3
"""Convert Claude Code sub-agent review trajectories to ShareGPT format.

Reads all sub-agent JSONL transcripts from a session and produces a
ShareGPT-format JSON file suitable for fine-tuning.

Usage:
    python scripts/convert_trajectories_to_sharegpt.py \
        --session-dir /path/to/session/subagents/ \
        --output data/review_trajectories_sharegpt.json
"""

import argparse
import json
from pathlib import Path


def extract_text(content) -> str:
    """Extract displayable text from a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if not isinstance(c, dict):
                continue
            typ = c.get("type", "")
            if typ == "text":
                parts.append(c.get("text", ""))
            elif typ == "tool_use":
                name = c.get("name", "unknown")
                inp = c.get("input", {})
                # Compact representation of tool calls
                if name == "Read":
                    parts.append(f"[Tool: Read {inp.get('file_path', '?')}]")
                elif name == "Write":
                    fp = inp.get("file_path", "?")
                    content_preview = inp.get("content", "")[:200]
                    parts.append(f"[Tool: Write {fp}]\n{content_preview}...")
                elif name == "Bash":
                    parts.append(f"[Tool: Bash] {inp.get('command', '?')}")
                elif name == "Glob":
                    parts.append(f"[Tool: Glob] {inp.get('pattern', '?')}")
                elif name == "Grep":
                    parts.append(f"[Tool: Grep] {inp.get('pattern', '?')}")
                elif "search_arxiv" in name or "arxiv" in name.lower():
                    parts.append(f"[Tool: search_arxiv] {inp.get('query', '?')}")
                elif name == "ToolSearch":
                    parts.append(f"[Tool: ToolSearch] {inp.get('query', '?')}")
                else:
                    parts.append(f"[Tool: {name}] {json.dumps(inp)[:200]}")
            elif typ == "tool_result":
                result = c.get("content", "")
                if isinstance(result, list):
                    result = " ".join(
                        r.get("text", "") for r in result if isinstance(r, dict)
                    )
                result_str = str(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "... [truncated]"
                parts.append(f"[Tool Result]: {result_str}")
            elif typ == "image":
                source = c.get("source", {})
                parts.append(f"[Image: {source.get('media_type', 'image')}]")
        return "\n".join(parts)
    return str(content)


def convert_transcript(jsonl_path: Path):
    """Convert a single sub-agent JSONL transcript to ShareGPT format."""
    messages = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "message" not in msg:
                continue
            role = msg["message"].get("role", "")
            content = msg["message"].get("content", "")
            if not role or not content:
                continue
            text = extract_text(content)
            if not text.strip():
                continue
            messages.append({"role": role, "text": text})

    if not messages:
        return None

    # Convert to ShareGPT format: merge consecutive same-role messages
    conversations = []
    current_role = None
    current_text = []

    for msg in messages:
        sharegpt_role = "human" if msg["role"] == "user" else "gpt"
        if sharegpt_role == current_role:
            current_text.append(msg["text"])
        else:
            if current_role is not None:
                conversations.append({
                    "from": current_role,
                    "value": "\n".join(current_text),
                })
            current_role = sharegpt_role
            current_text = [msg["text"]]

    if current_role is not None:
        conversations.append({
            "from": current_role,
            "value": "\n".join(current_text),
        })

    if not conversations:
        return None

    # Extract submission_id from the first message
    first_msg = conversations[0]["value"]
    submission_id = "unknown"
    for word in first_msg.split():
        if len(word) == 10 and word.isalnum():
            submission_id = word
            break
    # Try harder
    if submission_id == "unknown":
        import re
        match = re.search(r"paper\s+(\w{10,12})", first_msg)
        if match:
            submission_id = match.group(1)

    return {
        "id": submission_id,
        "conversations": conversations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, help="Path to subagents/ directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--full", action="store_true",
                        help="Include full tool results (default: truncate to 500 chars)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    jsonl_files = sorted(session_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} sub-agent transcripts")

    dataset = []
    for jsonl_path in jsonl_files:
        entry = convert_transcript(jsonl_path)
        if entry:
            n_turns = len(entry["conversations"])
            dataset.append(entry)

    print(f"Converted {len(dataset)} trajectories")

    # Stats
    total_turns = sum(len(e["conversations"]) for e in dataset)
    print(f"Total conversation turns: {total_turns}")
    print(f"Avg turns per trajectory: {total_turns / len(dataset):.1f}")

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
