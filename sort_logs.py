#!/usr/bin/env python3
"""Sort hand log lines in a log file by hand number."""

import re
import sys


def sort_log_file(path: str) -> None:
    with open(path) as f:
        lines = f.readlines()

    hand_pattern = re.compile(r"^Hand (\d+):")
    header_lines = []
    hand_lines = []
    footer_lines = []

    # Split into header (before hands), hand lines, and footer (after hands)
    section = "header"
    for line in lines:
        if section == "header":
            if hand_pattern.match(line):
                section = "hands"
                hand_lines.append(line)
            else:
                header_lines.append(line)
        elif section == "hands":
            if hand_pattern.match(line):
                hand_lines.append(line)
            else:
                section = "footer"
                footer_lines.append(line)
        else:
            footer_lines.append(line)

    # Sort hand lines by hand number
    hand_lines.sort(key=lambda l: int(hand_pattern.match(l).group(1)))

    # Renumber the (x/total) counters
    total = len(hand_lines)
    renumbered = []
    for i, line in enumerate(hand_lines, 1):
        line = re.sub(r"\(\d+/\d+\)", f"({i}/{total})", line)
        renumbered.append(line)

    with open(path, "w") as f:
        f.writelines(header_lines + renumbered + footer_lines)

    print(f"Sorted {len(renumbered)} hands in {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sort_logs.py <logfile>")
        sys.exit(1)
    sort_log_file(sys.argv[1])
