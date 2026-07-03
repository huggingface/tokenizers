#!/usr/bin/env python3
"""Insert or replace the pipeline-bench section in a PR body.

Usage: upsert_pr_section.py <current_body_file> <section_file>

Reads the current PR body and the rendered benchmark markdown, then writes an
updated body to stdout with the benchmark wrapped in a marker-delimited block.
Replaces the block in place if the markers are already present, otherwise
appends it — so the maintainer's own description is never clobbered.
"""
import re
import sys

START = "<!-- pipeline-bench:start -->"
END = "<!-- pipeline-bench:end -->"


def main():
    body = open(sys.argv[1]).read()
    section = open(sys.argv[2]).read().strip()
    block = f"{START}\n{section}\n{END}"
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if pattern.search(body):
        updated = pattern.sub(lambda _: block, body)
    elif body.strip():
        updated = f"{body.rstrip()}\n\n{block}\n"
    else:
        updated = f"{block}\n"
    sys.stdout.write(updated)


if __name__ == "__main__":
    main()
