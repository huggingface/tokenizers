"""
Helper to generate type stubs for python-only methods of the `Tokenizer` class

eg, in __init__.py:

```python
def from_pretrained(...):
    ...

Tokenizer.from_pretrained = staticmethod(from_pretrained)
```
"""

import ast
import copy
import textwrap
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "tokenizers"
SOURCE = PACKAGE / "__init__.py"
STUB = PACKAGE / "tokenizers.pyi"


def attached_staticmethods(module: ast.Module) -> list[tuple[str, str, ast.FunctionDef]]:
    """`(class, attribute, function)` for every `Class.attribute = staticmethod(function)`."""
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    attached = []
    for node in module.body:
        match node:
            case ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id=cls), attr=attribute)],
                value=ast.Call(func=ast.Name(id="staticmethod"), args=[ast.Name(id=function)]),
            ):
                attached.append((cls, attribute, functions[function]))
    return attached


def declaration(attribute: str, function: ast.FunctionDef) -> str:
    docstring = ast.get_docstring(function, clean=False)
    declared = copy.copy(function)
    declared.name = attribute
    declared.decorator_list = [ast.Name("staticmethod")]
    declared.body = [ast.Expr(ast.Constant(... if docstring is None else docstring))]
    return ast.unparse(declared)


def annotation_names(function: ast.FunctionDef) -> set[str]:
    annotations = [arg.annotation for arg in ast.walk(function.args) if isinstance(arg, ast.arg)]
    annotations.append(function.returns)
    return {
        node.id
        for annotation in annotations
        if annotation is not None
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name)
    }


def bound_by_imports(module: ast.Module) -> dict[str, str]:
    """Each name an absolute import binds, mapped to a statement importing that name alone."""
    bound = {}
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                bound[alias.asname or alias.name] = ast.unparse(ast.ImportFrom(node.module, [alias], 0))
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.asname or alias.name.partition(".")[0]] = ast.unparse(ast.Import([alias]))
    return bound


def bound_names(module: ast.Module) -> set[str]:
    defined = {node.name for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    return defined | set(bound_by_imports(module))


def with_member(stub: str, cls: str, member: str) -> str:
    module = ast.parse(stub)
    [class_def] = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name == cls]
    lines = stub.splitlines(keepends=True)
    end = class_def.end_lineno
    return "".join(lines[:end]) + textwrap.indent(member, "    ") + "\n" + "".join(lines[end:])


def with_imports(stub: str, imports: list[str]) -> str:
    module = ast.parse(stub)
    ends = [node.end_lineno for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    last = max(end for end in ends if end is not None)
    lines = stub.splitlines(keepends=True)
    return "".join(lines[:last]) + "".join(statement + "\n" for statement in imports) + "".join(lines[last:])


def main() -> None:
    source = ast.parse(SOURCE.read_text())
    source_imports = bound_by_imports(source)
    stub = STUB.read_text()
    for cls, attribute, function in attached_staticmethods(source):
        stub = with_member(stub, cls, declaration(attribute, function))
        missing = annotation_names(function) - bound_names(ast.parse(stub))
        stub = with_imports(stub, [source_imports[name] for name in sorted(missing) if name in source_imports])
    STUB.write_text(stub)
    print(f"declared {len(attached_staticmethods(source))} Python-attached member(s) in {STUB}")


if __name__ == "__main__":
    main()
