import argparse
import inspect
import os
from pathlib import Path


INDENT = " " * 4
GENERATED_COMMENT = "# Generated content DO NOT EDIT\n"

OVERRIDES = {
    ("tokenizers", "AddedToken", "__init__"): "(self, content=None, single_word=False, lstrip=False, rstrip=False, normalized=True, special=False)",
    ("tokenizers.decoders", "Strip", "__init__"): "(self, content=' ', left=0, right=0)",
    ("tokenizers.processors", "TemplateProcessing", "__init__"): "(self, single=None, pair=None, special_tokens=None)",
}


def do_indent(text: str, indent: str):
    return text.replace("\n", f"\n{indent}")


def function(obj, indent, text_signature=None, owner=None):
    name = obj.__name__

    # 1) Figure out a usable text_signature
    if text_signature is None:
        text_signature = getattr(obj, "__text_signature__", None)
    if owner is not None:
        key = (getattr(owner, "__module__", ""), owner.__name__, name)
        if key in OVERRIDES:
            text_signature = OVERRIDES[key]
    if text_signature is None:
        text_signature = "()"
    else:
        text_signature = text_signature.replace("$self", "self").replace(" /,", "")

    if name in ("__getitem__", "__setitem__"):
        # Always expose magic indexing methods, even if they lack a __text_signature__
        # (PyO3 magic methods often do).
        if name == "__getitem__":
            text_signature = "(self, key)"
        else:
            text_signature = "(self, key, value)"

    # 2) Safely handle missing docstrings
    doc = obj.__doc__ or ""

    string = ""
    string += f"{indent}def {name}{text_signature}:\n"
    indent += INDENT
    string += f'{indent}"""\n'
    if doc:
        string += f"{indent}{do_indent(doc, indent)}\n"
    string += f'{indent}"""\n'
    string += f"{indent}pass\n"
    string += "\n\n"
    return string


def member_sort(member):
    if inspect.isclass(member):
        value = 10 + len(inspect.getmro(member))
    else:
        value = 1
    return value


def fn_predicate(obj):
    always = {"__getitem__", "__setitem__", "__getstate__", "__setstate__", "__getnewargs__"}
    if inspect.ismethoddescriptor(obj) or inspect.isbuiltin(obj):
        name = obj.__name__
        # Always expose magic indexing methods, even if they start with "_"
        # or lack a __text_signature__ (PyO3 magic methods often do).
        if name in always:
            return True
        return obj.__text_signature__ and not obj.__name__.startswith("_")

    if inspect.isgetsetdescriptor(obj):
        return not obj.__name__.startswith("_")
    return False


def get_module_members(module):
    members = [
        member
        for name, member in inspect.getmembers(module)
        if not name.startswith("_") and not inspect.ismodule(member)
    ]
    members.sort(key=member_sort)
    return members


def pyi_file(obj, indent="", owner=None):
    string = ""
    if inspect.ismodule(obj):
        string += GENERATED_COMMENT
        members = get_module_members(obj)
        for member in members:
            string += pyi_file(member, indent)

    elif inspect.isclass(obj):
        indent += INDENT
        mro = inspect.getmro(obj)
        if len(mro) > 2:
            inherit = f"({mro[1].__name__})"
        else:
            inherit = ""
        string += f"class {obj.__name__}{inherit}:\n"

        body = ""
        if obj.__doc__:
            body += f'{indent}"""\n{indent}{do_indent(obj.__doc__, indent)}\n{indent}"""\n'

        fns = inspect.getmembers(obj, fn_predicate)

        # Init
        if obj.__text_signature__:
            init_sig = OVERRIDES.get((obj.__module__, obj.__name__, "__init__"), obj.__text_signature__)
            init_sig = init_sig.replace("$self", "self").replace(" /,", "")
            body += f"{indent}def __init__{init_sig}:\n"
            body += f"{indent + INDENT}pass\n"
            body += "\n"

        for name, fn in fns:
            body += pyi_file(fn, indent=indent, owner=obj)

        if not body:
            body += f"{indent}pass\n"

        string += body
        string += "\n\n"

    elif inspect.isbuiltin(obj):
        string += f"{indent}@staticmethod\n"
        string += function(obj, indent, owner=owner)

    elif inspect.ismethoddescriptor(obj):
        string += function(obj, indent, owner=owner)

    elif inspect.isgetsetdescriptor(obj):
        string += f"{indent}@property\n"
        string += function(obj, indent, text_signature="(self)", owner=owner)
        # Expose setter in stubs for properties that are writable in Python.
        # If a descriptor is actually read-only at runtime, type checkers may still allow
        # assignment but the runtime will raise, which is acceptable for stubs.
        string += f"{indent}@{obj.__name__}.setter\n"
        string += function(obj, indent, text_signature="(self, value)", owner=owner)
    else:
        raise Exception(f"Object {obj} is not supported")
    return string


def py_file(module, origin):
    members = get_module_members(module)

    string = GENERATED_COMMENT
    string += f"from .. import {origin}\n"
    string += "\n"
    for member in members:
        name = member.__name__
        string += f"{name} = {origin}.{name}\n"
    return string


import subprocess
from typing import List, Optional, Tuple


def do_ruff(code, is_pyi: bool):
    command = ["ruff", "format", "--config", "pyproject.toml"]
    command.extend(["--stdin-filename", "test.pyi" if is_pyi else "test.py", "-"])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, stderr = process.communicate(input=code.encode("utf-8"))
    if stderr:
        print(code)
        print(f"Ruff error: {stderr.decode('utf-8')}")
        return code
    return stdout.decode("utf-8")


def write(module, directory, origin, check=False):
    submodules = [(name, member) for name, member in inspect.getmembers(module) if inspect.ismodule(member)]

    filename = os.path.join(directory, "__init__.pyi")
    pyi_content = pyi_file(module)

    # Inject extra hints for hand-written Python modules layered on top of the extension.
    if origin == "tokenizers":
        extra = """
from enum import Enum
from typing import List, Tuple, Union, Any

Offsets = Tuple[int, int]
TextInputSequence = str
PreTokenizedInputSequence = Union[List[str], Tuple[str, ...]]
TextEncodeInput = Union[
    TextInputSequence,
    Tuple[TextInputSequence, TextInputSequence],
    List[TextInputSequence],
]
PreTokenizedEncodeInput = Union[
    PreTokenizedInputSequence,
    Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence],
    List[PreTokenizedInputSequence],
]
InputSequence = Union[TextInputSequence, PreTokenizedInputSequence]
EncodeInput = Union[TextEncodeInput, PreTokenizedEncodeInput]


class OffsetReferential(Enum):
    ORIGINAL = "original"
    NORMALIZED = "normalized"


class OffsetType(Enum):
    BYTE = "byte"
    CHAR = "char"


class SplitDelimiterBehavior(Enum):
    REMOVED = "removed"
    ISOLATED = "isolated"
    MERGED_WITH_PREVIOUS = "merged_with_previous"
    MERGED_WITH_NEXT = "merged_with_next"
    CONTIGUOUS = "contiguous"

from .implementations import (
    BertWordPieceTokenizer,
    ByteLevelBPETokenizer,
    CharBPETokenizer,
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer,
)

def __getattr__(name: str) -> Any: ...
BertWordPieceTokenizer: Any
ByteLevelBPETokenizer: Any
CharBPETokenizer: Any
SentencePieceBPETokenizer: Any
SentencePieceUnigramTokenizer: Any
"""
        pyi_content += extra

    if origin == "normalizers":
        pyi_content += """
from typing import Dict

NORMALIZERS: Dict[str, Normalizer]

def unicode_normalizer_from_str(normalizer: str) -> Normalizer: ...
"""

    try:
        pyi_content = do_ruff(pyi_content, is_pyi=True)
    except Exception as e:
        print(f"Ruff error: {e}")

    os.makedirs(directory, exist_ok=True)
    if check:
        with open(filename, "r") as f:
            data = f.read()
            assert data == pyi_content, f"The content of {filename} seems outdated, please run `python stub.py`"
    else:
        with open(filename, "w") as f:
            f.write(pyi_content)

    filename = os.path.join(directory, "__init__.py")
    py_content = py_file(module, origin)
    try:
        py_content = do_ruff(py_content, is_pyi=False)
    except Exception as e:
        print(f"Ruff error: {e}")

    os.makedirs(directory, exist_ok=True)

    is_auto = False
    if not os.path.exists(filename):
        is_auto = True
    else:
        with open(filename, "r") as f:
            line = f.readline()
            if line == GENERATED_COMMENT:
                is_auto = True

    if is_auto:
        if check:
            with open(filename, "r") as f:
                data = f.read()
                assert data == py_content, f"The content of {filename} seems outdated, please run `python stub.py`"
        else:
            with open(filename, "w") as f:
                f.write(py_content)

    for name, submodule in submodules:
        write(submodule, os.path.join(directory, name), f"{name}", check=check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()
    import tokenizers

    # `tokenizers.tokenizers` is the extension module; attribute access is dynamic.
    write(tokenizers.tokenizers, "py_src/tokenizers/", "tokenizers", check=args.check)  # type: ignore[attr-defined]
