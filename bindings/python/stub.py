import argparse
import inspect
import os
from pathlib import Path

import black


INDENT = " " * 4
GENERATED_COMMENT = "# Generated content DO NOT EDIT\n"


def do_indent(text: str, indent: str):
    return text.replace("\n", f"\n{indent}")


def function(obj, indent, text_signature=None):
    if text_signature is None:
        text_signature = obj.__text_signature__
    string = ""
    string += f"{indent}def {obj.__name__}{text_signature}:\n"
    indent += INDENT
    string += f'{indent}"""\n'
    string += f"{indent}{do_indent(obj.__doc__, indent)}\n"
    string += f'{indent}"""\n'
    string += f"{indent}pass\n"
    string += "\n"
    string += "\n"
    return string


def member_sort(member):
    if inspect.isclass(member):
        value = 10 + len(inspect.getmro(member))
    else:
        value = 1
    return value


def fn_predicate(obj):
    value = inspect.ismethoddescriptor(obj) or inspect.isbuiltin(obj)
    if value:
        return obj.__doc__ and obj.__text_signature__ and not obj.__name__.startswith("_")
    if inspect.isgetsetdescriptor(obj):
        return obj.__doc__ and not obj.__name__.startswith("_")
    return False


def get_module_members(module):
    members = [
        member
        for name, member in inspect.getmembers(module)
        if not name.startswith("_") and not inspect.ismodule(member)
    ]
    members.sort(key=member_sort)
    return members


def pyi_file(obj, indent=""):
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
            body += f"{indent}def __init__{obj.__text_signature__}:\n"
            body += f"{indent+INDENT}pass\n"
            body += "\n"

        for (name, fn) in fns:
            body += pyi_file(fn, indent=indent)

        if not body:
            body += f"{indent}pass\n"

        string += body
        string += "\n\n"

    elif inspect.isbuiltin(obj):
        string += f"{indent}@staticmethod\n"
        string += function(obj, indent)

    elif inspect.ismethoddescriptor(obj):
        string += function(obj, indent)

    elif inspect.isgetsetdescriptor(obj):
        # TODO it would be interesing to add the setter maybe ?
        string += f"{indent}@property\n"
        string += function(obj, indent, text_signature="(self)")
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


def do_black(content, is_pyi):
    mode = black.Mode(
        target_versions={black.TargetVersion.PY35},
        line_length=119,
        is_pyi=is_pyi,
        string_normalization=True,
        experimental_string_processing=False,
    )
    try:
        return black.format_file_contents(content, fast=True, mode=mode)
    except black.NothingChanged:
        return content


def write(module, directory, origin, check=False):
    submodules = [(name, member) for name, member in inspect.getmembers(module) if inspect.ismodule(member)]

    filename = os.path.join(directory, "__init__.pyi")
    pyi_content = pyi_file(module)
    pyi_content = do_black(pyi_content, is_pyi=True)
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
    py_content = do_black(py_content, is_pyi=False)
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

    write(tokenizers.tokenizers, "py_src/tokenizers/", "tokenizers", check=args.check)
