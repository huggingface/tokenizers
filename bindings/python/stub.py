import argparse
import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path


GENERATED_COMMENT = "# Generated content DO NOT EDIT\n"


def public_members(module):
    return [
        member
        for name, member in inspect.getmembers(module)
        if not name.startswith("_") and not inspect.ismodule(member)
    ]


def forwarder(module, origin):
    members = public_members(module)
    lines = [GENERATED_COMMENT, f"from .. import {origin}", ""]
    for member in members:
        if getattr(member, "__module__", "") == "typing":
            continue
        member_name = getattr(member, "__name__", None)
        if member_name:
            lines.append(f"{member_name} = {origin}.{member_name}")
    lines.append("")
    return "\n".join(lines)


def do_ruff(code):
    command = ["ruff", "format", "--config", "pyproject.toml", "--stdin-filename", "__init__.py", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, stderr = process.communicate(input=code.encode("utf-8"))
    if stderr:
        print(code)
        print(f"Ruff error: {stderr.decode('utf-8')}")
        return code
    return stdout.decode("utf-8")


def format_docstring(docstring: str, indent: int = 4) -> str:
    """Format a docstring for insertion into a .pyi file."""
    if not docstring:
        return ""

    indent_str = " " * indent
    lines = docstring.strip().split("\n")

    if len(lines) == 1:
        # Single line docstring
        return f'{indent_str}"""{lines[0]}"""\n'

    # Multi-line docstring
    result = [f'{indent_str}"""']
    result.extend(indent_str + line.rstrip() for line in lines)
    result.append(f'{indent_str}"""')
    return "\n".join(result) + "\n"


def get_module(module_name: str):
    """Get module by name, handling tokenizers submodules."""
    import tokenizers
    if module_name == "tokenizers":
        return tokenizers
    return getattr(tokenizers, module_name.split(".")[-1], None)


def add_docstring_to_stub(line: str, docstring: str, indent: int) -> str:
    """Convert 'def foo(): ...' or multi-line ending with '...' to include docstring."""
    if line.rstrip().endswith('...'):
        base = line.rstrip()[:-3]  # Remove ...
        if not base.rstrip().endswith(':'):
            base = base.rstrip() + ':'
        inner = ' ' * (indent + 4)
        return f"{base}\n{inner}\"\"\"{docstring}\"\"\"\n{inner}..."
    return line


def add_docstrings_to_pyi(pyi_file: Path, module_name: str):
    """Add docstrings from the actual module to a .pyi file."""
    module = get_module(module_name)
    if module is None:
        print(f"Could not find module {module_name}")
        return

    content = pyi_file.read_text()
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Could not parse {pyi_file}: {e}")
        return

    lines = content.splitlines(keepends=True)

    # Collect insertions: (start_line, end_line, docstring, indent)
    # start_line and end_line are 0-indexed
    insertions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            obj = getattr(module, node.name, None)
            if obj and getattr(obj, "__doc__", None):
                indent = len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())
                insertions.append((node.lineno - 1, node.lineno - 1, obj.__doc__.strip(), indent))

            # Process methods and properties within the class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_obj = getattr(module, node.name, None)
                    if not class_obj:
                        continue
                    # For properties, get descriptor; for methods, get method
                    method_obj = getattr(class_obj, item.name, None)
                    if method_obj and getattr(method_obj, "__doc__", None):
                        start = item.lineno - 1
                        end = (item.end_lineno - 1) if item.end_lineno else start
                        indent = len(lines[start]) - len(lines[start].lstrip())
                        insertions.append((start, end, method_obj.__doc__.strip(), indent))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only process top-level functions (not nested in classes)
            # Check if this is a top-level node
            if node in tree.body:
                obj = getattr(module, node.name, None)
                if obj and getattr(obj, "__doc__", None):
                    start = node.lineno - 1
                    end = (node.end_lineno - 1) if node.end_lineno else start
                    indent = len(lines[start]) - len(lines[start].lstrip())
                    insertions.append((start, end, obj.__doc__.strip(), indent))

    # Sort by start line descending to apply from bottom to top
    insertions.sort(key=lambda x: x[0], reverse=True)

    # Apply insertions
    for start, end, docstring, indent in insertions:
        # Get all lines for this definition
        def_lines = [lines[i].rstrip('\n') for i in range(start, end + 1)]
        combined = '\n'.join(def_lines)

        # Check if it ends with ... (stub pattern)
        if combined.rstrip().endswith('...'):
            inner_indent = ' ' * (indent + 4)
            formatted_doc = format_docstring(docstring, indent=indent + 4).rstrip('\n')

            if len(def_lines) == 1:
                # Single line: def foo(): ...
                base = def_lines[0].rstrip()[:-3].rstrip()
                if not base.endswith(':'):
                    base += ':'
                new_content = f"{base}\n{formatted_doc}\n{inner_indent}...\n"
            else:
                # Multi-line signature
                last_line = def_lines[-1].rstrip()[:-3].rstrip()  # Remove ...
                new_lines = def_lines[:-1] + [last_line]
                new_content = '\n'.join(new_lines) + '\n' + formatted_doc + '\n' + inner_indent + '...\n'

            # Replace lines[start:end+1] with new_content
            lines[start:end + 1] = [new_content]

    # Write back
    pyi_file.write_text(''.join(lines))
    print(f"Added docstrings to {pyi_file}")


def write(module, directory, origin, check=False):
    submodules = [
        (name, member)
        for name, member in inspect.getmembers(module)
        if inspect.ismodule(member)
    ]
    os.makedirs(directory, exist_ok=True)

    filename = os.path.join(directory, "__init__.py")
    py_content = forwarder(module, origin)
    try:
        py_content = do_ruff(py_content)
    except Exception as e:
        print(f"Ruff error: {e}")

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

    print(f"Wrote stub for module: {origin}, submodules: {[name for name, _ in submodules]}")
    for name, submodule in submodules:
        try:
            write(submodule, os.path.join(directory, name), f"{name}", check=check)
        except Exception as e:
            print(f"Something went wrong with {name}, {submodule}: {e}")

def process_all_pyi_files(base_dir: str):
    """Process all .pyi files in the directory tree to add docstrings."""
    base_path = Path(base_dir)

    # Map relative paths to module names
    for pyi_file in base_path.rglob("*.pyi"):
        # Skip __init__.pyi as it typically just has imports
        if pyi_file.name == "__init__.pyi" and pyi_file.parent.name == "tokenizers":
            # Process the main tokenizers module
            add_docstrings_to_pyi(pyi_file, "tokenizers")
        elif pyi_file.name == "__init__.pyi":
            # Skip other __init__.pyi files
            continue
        elif pyi_file.name == "tokenizers.pyi":
            # Skip the tokenizers.pyi file as it's just imports
            continue
        else:
            # Convert file path to module name
            # e.g., py_src/tokenizers/decoders.pyi -> tokenizers.decoders
            rel_path = pyi_file.relative_to(base_path)
            # rel_path will be something like "decoders.pyi" or "subdir/file.pyi"
            # We want to convert to "tokenizers.decoders" or "tokenizers.subdir.file"
            parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            if parts:
                module_name = "tokenizers." + ".".join(parts)
            else:
                module_name = "tokenizers"

            add_docstrings_to_pyi(pyi_file, module_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()
    import tokenizers

    write(tokenizers, "py_src/tokenizers/", "tokenizers", check=args.check)  # type: ignore[attr-defined]

    # Process all .pyi files to add docstrings
    if not args.check:
        print("\nAdding docstrings to .pyi files...")
        process_all_pyi_files("py_src/tokenizers")
