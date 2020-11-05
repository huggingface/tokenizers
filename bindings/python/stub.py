import inspect
import os
import black
from pathlib import Path

INDENT = " " * 4


def do_indent(text: str, indent: str):
    return text.replace("\n", f"\n{indent}")


def function(obj, indent):
    string = ""
    string += f"{indent}def {obj.__name__}{obj.__text_signature__}:\n"
    indent += INDENT
    string += f'{indent}"""\n'
    string += f"{indent}{do_indent(obj.__doc__, indent)}\n"
    string += f'{indent}"""\n'
    string += f"{indent}pass\n"
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

    return value and obj.__doc__ and obj.__text_signature__ and not obj.__name__.startswith("_")


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
        string += "#Generated content DO NOT EDIT\n"
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

    else:
        raise Exception(f"Object {obj} is not supported")
    return string


def py_file(module, origin):
    members = get_module_members(module)

    string = "#Generated content DO NOT EDIT\n"
    for member in members:
        string += f"from {origin} import {member.__name__}\n"
    return string


def do_black(filename, is_pyi):
    filename = Path(filename)
    mode = black.Mode(
        target_versions={black.TargetVersion.PY36},
        line_length=100,
        is_pyi=False,
        string_normalization=True,
        experimental_string_processing=False,
    )
    check = False
    diff = False
    color = False
    quiet = True
    verbose = False
    report = black.Report(check=check, diff=diff, quiet=quiet, verbose=verbose)
    write_back = black.WriteBack.from_configuration(check=check, diff=diff, color=color)
    black.reformat_one(filename, fast=True, write_back=write_back, mode=mode, report=report)


def write(module, directory, origin):
    submodules = [
        (name, member) for name, member in inspect.getmembers(module) if inspect.ismodule(member)
    ]

    filename = os.path.join(directory, "__init__.pyi")
    pyi_content = pyi_file(module)
    os.makedirs(directory, exist_ok=True)
    with open(filename, "w") as f:
        f.write(pyi_content)
    do_black(filename, is_pyi=True)

    filename = os.path.join(directory, "__init__.py")
    py_content = py_file(module, origin)
    os.makedirs(directory, exist_ok=True)
    with open(filename, "w") as f:
        f.write(py_content)

    do_black(filename, is_pyi=False)

    for name, submodule in submodules:
        write(submodule, os.path.join(directory, name), f"{origin}.{name}")


if __name__ == "__main__":
    import tokenizers

    write(tokenizers.tokenizers, "py_src/tokenizers/", "tokenizers.tokenizers")
