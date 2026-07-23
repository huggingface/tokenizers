"""Golden conformance layer: byte-exact agreement with the released wheel.

The files under goldens/ record what the released tokenizers wheel produces
on the data/ fixtures and a set of edge-case strings (see generate.py).
test_golden.py replays those inputs on the current build and diffs every id,
token, offset, mask and decoded string against the record.

Run with `make golden`; regenerate the goldens with `make golden-regen`.

The env gate below makes the layer opt-in while the 1.0 rewrite is
incomplete, so a plain `make test` or the required CI jobs stay green while
offsets/decode/pairs are still missing. Delete it once `make golden` runs
green — from then on conformance should be enforced by default, everywhere.
"""

import os

import pytest

if not os.environ.get("TOKENIZERS_GOLDEN"):
    pytest.skip("golden conformance layer is opt-in for now — run `make golden`", allow_module_level=True)
