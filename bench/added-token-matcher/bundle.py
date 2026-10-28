#!/usr/bin/env python3
"""Write start_job.sh: start_job.in.sh with the harness, corpora.py and this worktree's diff against MAIN embedded,
so the job command is that one file and the branch is measured without pushing it."""
import pathlib
import subprocess
import sys

MAIN = "d9a5e8af4b5898aff65981ab472d5466a43698bc"
here = pathlib.Path(__file__).parent
repo = here.parent.parent


def git(*args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


untracked = git("ls-files", "--others", "--exclude-standard", "tokenizers").split()
if untracked:
    sys.exit(f"untracked files would be missing from the diff, `git add -N` them: {untracked}")
diff = git("diff", MAIN, "--", "tokenizers")
dirty = "+dirty" if git("status", "--porcelain", "tokenizers") else ""
ref = f"{git('rev-parse', '--abbrev-ref', 'HEAD').strip()} @ {git('rev-parse', '--short', 'HEAD').strip()}{dirty}"

script = (here / "start_job.in.sh").read_text()
for mark, text in [("@@CORPORA_PY@@", (here / "corpora.py").read_text()),
                   ("@@CARGO_TOML@@", (here / "harness" / "Cargo.toml").read_text()),
                   ("@@MAIN_RS@@", (here / "harness" / "src" / "main.rs").read_text()),
                   ("@@BRANCH_DIFF@@", diff),
                   ("@@BRANCH_REF@@", ref)]:
    script = script.replace(mark, text.rstrip("\n"))
(here / "branch.diff").write_text(diff)
(here / "start_job.sh").write_text(script)
print(f"wrote start_job.sh ({len(script)} bytes), branch {ref}, diff {len(diff)} bytes")
