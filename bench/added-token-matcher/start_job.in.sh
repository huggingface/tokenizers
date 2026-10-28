#!/usr/bin/env bash
# Added-token matcher: main (bucket matcher) vs main + branch.diff (aho-corasick), the matcher alone and the full
# encode, on 20 MB English, code and agentic corpora. One thread, arms interleaved, ROUNDS rounds.
# Self-contained; runs in rust:1.93.0-bookworm. Generated from start_job.in.sh by bundle.py: edit that, then `make bundle`.
set -euo pipefail
MAIN=${MAIN:-d9a5e8af4b5898aff65981ab472d5466a43698bc}
ROUNDS=${ROUNDS:-5}
SECS=${SECS:-1}
say() { echo "== $(date -u +%T) $*"; }
steal() { awk '/^cpu /{print $9}' /proc/stat; }

say "start rounds=$ROUNDS secs=$SECS flavor=${FLAVOR:-?}"
CPU0=$(grep Cpus_allowed_list /proc/self/status | cut -f2 | cut -d, -f1 | cut -d- -f1)
PIN=""; command -v taskset >/dev/null && taskset -c "$CPU0" true 2>/dev/null && PIN="taskset -c $CPU0"
echo "== machine $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), nproc $(nproc), kernel $(uname -r), pinned to ${PIN:-none}"

say "corpora"
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
cat > /corpora.py <<'PY'
@@CORPORA_PY@@
PY
uv run --quiet /corpora.py /corpora
wc -c /corpora/*.txt

say "checkouts @ $MAIN"
git clone -q --filter=blob:none https://github.com/huggingface/tokenizers /src
git -C /src worktree add -q /w/main "$MAIN"
git -C /src worktree add -q /w/branch "$MAIN"
cat > /branch.diff <<'MATCHER_PATCH_EOF'
@@BRANCH_DIFF@@
MATCHER_PATCH_EOF
git -C /w/branch apply /branch.diff
echo "== branch: main + branch.diff ($(grep -c '^+' /branch.diff) added lines): @@BRANCH_REF@@"

build() {  # arm checkout cargo-flags
  mkdir -p /h/$1/src
  cat > /h/$1/Cargo.toml <<'TOML'
@@CARGO_TOML@@
TOML
  sed -i "s|/w/branch|$2|g" /h/$1/Cargo.toml
  cat > /h/$1/src/main.rs <<'RS'
@@MAIN_RS@@
RS
  cp $2/tokenizers/Cargo.lock /h/$1/
  (cd /h/$1 && cargo build -q --release $3)
  echo "== $1 Cargo.lock: $(grep -A1 -E 'name = "(aho-corasick|memchr)"' /h/$1/Cargo.lock | grep version | tr '\n' ' ')"
}
say "build"
build main /w/main "--features buckets"
build branch /w/branch ""

s0=$(steal); say "rounds"
for round in $(seq 1 "$ROUNDS"); do
  if [ $((round % 2)) = 1 ]; then order="main branch"; else order="branch main"; fi
  for mode in matcher e2e; do
    for arm in $order; do $PIN /h/$arm/target/release/matcher-bench $mode "$SECS" /corpora/sets.tsv "$round"; done
  done
  say "round $round done"
done
say "end, steal ticks $(( $(steal) - s0 ))"
