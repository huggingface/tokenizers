import argparse
import inspect
import os
import subprocess


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

    for name, submodule in submodules:
        print(f"Writing stub for submodule: {name}")
        write(submodule, os.path.join(directory, name), f"{name}", check=check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()
    import tokenizers

    write(tokenizers, "py_src/tokenizers/", "tokenizers", check=args.check)  # type: ignore[attr-defined]
