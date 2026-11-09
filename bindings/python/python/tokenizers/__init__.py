"""
Python bindings for Hugging Face's Tokenizers rust library

Encode text to token ids and decode token ids back to text
"""

from . import decoders, models, normalizers, pre_tokenizers, processors
from .tokenizers import (
    CompiledTokenizer,
    Component,
    Encoding,
    Padding,
    Regex,
    Tokenizer,
    Truncation,
    __version__,
    get_max_threads,
    set_max_threads,
)

__all__ = [
    "CompiledTokenizer",
    "Component",
    "Encoding",
    "Padding",
    "Regex",
    "Tokenizer",
    "Truncation",
    "__version__",
    "get_max_threads",
    "set_max_threads",
    "decoders",
    "models",
    "normalizers",
    "pre_tokenizers",
    "processors",
]

# A multiprocessing worker can't know how many siblings share the cores: it defaults to 1 thread.
# Forked children already do (Rust side); spawn / forkserver workers have `multiprocessing` loaded
# by their bootstrap, so checking `sys.modules` costs nothing elsewhere.
import sys as _sys

from .tokenizers import _init_threads

_mp = _sys.modules.get("multiprocessing")
_init_threads(_mp is not None and _mp.parent_process() is not None)
del _sys, _mp, _init_threads
