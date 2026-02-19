#!/usr/bin/env python3
"""Extract individual experiment table sections from the formatted output."""

import re
import sys

input_text = open("/tmp/notion_tables.txt").read()

# Split by experiment headers
experiments = re.split(r"\n={80}\nEXPERIMENT: (\S+)\n={80}\n", input_text)

# experiments[0] is empty (before first match), then pairs of (name, content)
for i in range(1, len(experiments), 2):
    name = experiments[i]
    content = experiments[i + 1]

    # Extract status table
    status_match = re.search(r"--- STATUS TABLE ---\n(.*?)(?=\n--- PER-YEAR)", content, re.DOTALL)
    year_match = re.search(r"--- PER-YEAR RESULTS TABLE ---\n(.*?)$", content, re.DOTALL)

    if status_match:
        outfile = f"/tmp/notion_{name}_status.txt"
        with open(outfile, "w") as f:
            f.write(status_match.group(1).strip())
        print(f"Wrote {outfile}")

    if year_match:
        outfile = f"/tmp/notion_{name}_peryear.txt"
        with open(outfile, "w") as f:
            f.write(year_match.group(1).strip())
        print(f"Wrote {outfile}")
