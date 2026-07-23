"""Let transformers import with the in-tree tokenizers build installed.

transformers 5.x refuses to import unless the installed tokenizers version
satisfies the range in its dependency_versions_table.py (checked at import
time), which rejects our 1.0.0-dev build. This rewrites that one entry in the
venv's copy of the table to an unversioned "tokenizers". Run it once per
venv, after installing transformers; `make e2e` does.

The file is located without importing transformers: importing it would run
the very check being lifted.
"""

import importlib.util
import pathlib
import re

table = pathlib.Path(importlib.util.find_spec("transformers").origin).with_name("dependency_versions_table.py")
table.write_text(re.sub(r'"tokenizers": "tokenizers[^"]*"', '"tokenizers": "tokenizers"', table.read_text()))
print(f"tokenizers pin relaxed in {table}")
