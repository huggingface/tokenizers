#!/usr/bin/env bash
#
# Feature-matrix check for tk-encode.
#
# tk-encode's per-model features (`unigram`, `wordpiece`, `wordlevel`) and its
# `normalizers` feature are `cfg` gates on enum variants and match arms, not
# merely unused code. That means a combination can stop compiling while the
# default and all-features builds both stay green — which is exactly what CI's
# `--all-features` clippy job cannot see. Two real examples from the split:
#
#   * a match gained an unreachable `_ => ()` arm, which is a warning (and so a
#     `-D warnings` failure) only in the build where every variant is present;
#   * a `use Model as _` import was needed by two of the three model gates, so
#     it was an unused import in the third.
#
# So each configuration gets checked, linted and tested on its own.
#
# Per configuration:
#   cargo check  --all-targets
#   cargo clippy --all-targets -- -D warnings
#   cargo test   --lib --doc                  (the test count is reported)
#
# Usage:
#   scripts/feature_matrix.sh
#
# Exits non-zero if any configuration fails any of the three.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The workspace root, i.e. <repo>/tokenizers.
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PACKAGE="tk-encode"

# How much of a failing cargo log to echo inline. The interesting part -- the
# error, the failure list, the `test result:` line -- is always at the end.
LOG_TAIL_LINES=40

# label<TAB>cargo feature arguments
#
# Every entry carries at least one argument on purpose: bash 3.2, which is what
# macOS still ships as /bin/bash, treats the expansion of an empty array under
# `set -u` as an unbound variable. `--features bpe` is the default set spelled
# out, since `default = ["bpe"]`.
MATRIX=(
  $'no-default-features\t--no-default-features'
  $'default (bpe)\t--features bpe'
  $'bpe + unigram\t--features unigram'
  $'bpe + wordpiece\t--features wordpiece'
  $'bpe + wordlevel\t--features wordlevel'
  $'bpe + normalizers\t--features normalizers'
  $'bpe + all three models\t--features unigram,wordpiece,wordlevel'
  $'all-features\t--all-features'
)

if [ "$#" -gt 0 ]; then
  case "$1" in
    -h | --help)
      # The header comment block IS the help text; print it rather than keeping
      # a second copy in sync.
      awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "feature_matrix.sh: unexpected argument '$1'" >&2
      exit 2
      ;;
  esac
fi

cd "${WORKSPACE_DIR}"

LOG_DIR="$(mktemp -d)"
trap 'rm -rf "${LOG_DIR}"' EXIT

TOTAL="${#MATRIX[@]}"
FAILED=0

# Parallel arrays indexed by rung; bash 3.2 has no associative arrays.
res_label=()
res_check=()
res_clippy=()
res_test=()
res_count=()

# Runs one cargo invocation, returning its exit status.
#
# The output goes to a file rather than through a pipe: `cargo ... | grep ...`
# reports grep's status, so a failing cargo hides behind a successful filter.
# The caller reads the status directly and only greps the file afterwards.
run_step() {
  local log="$1"
  shift
  set +e
  "$@" > "${log}" 2>&1
  local rc=$?
  set -e
  return "${rc}"
}

# Sums the `test result: ok. N passed; ...` lines in a cargo test log.
test_count() {
  awk '/^test result:/ { for (i = 1; i <= NF; i++) if ($(i + 1) == "passed;") s += $i } END { print s + 0 }' "$1"
}

echo "feature matrix: ${TOTAL} configurations of ${PACKAGE}"
echo "                each runs cargo check + clippy -D warnings + test --lib --doc"
echo

start_all=$(date +%s)

for i in "${!MATRIX[@]}"; do
  rung="${MATRIX[${i}]}"
  label="${rung%%$'\t'*}"
  features="${rung#*$'\t'}"
  n=$((i + 1))

  read -r -a feature_args <<< "${features}"

  check_log="${LOG_DIR}/${n}-check.log"
  clippy_log="${LOG_DIR}/${n}-clippy.log"
  test_log="${LOG_DIR}/${n}-test.log"
  doc_log="${LOG_DIR}/${n}-doc.log"

  check="ok"
  clippy="ok"
  test_st="ok"
  count="-"
  rung_failed=0

  run_step "${check_log}" \
    cargo check -p "${PACKAGE}" --all-targets "${feature_args[@]}" \
    || {
      check="FAIL"
      rung_failed=1
    }

  run_step "${clippy_log}" \
    cargo clippy -p "${PACKAGE}" --all-targets "${feature_args[@]}" -- -D warnings \
    || {
      clippy="FAIL"
      rung_failed=1
    }

  run_step "${test_log}" \
    cargo test -p "${PACKAGE}" --lib "${feature_args[@]}" \
    || {
      test_st="FAIL"
      rung_failed=1
    }

  # Doc tests get their own run: cargo refuses `--doc` next to any other target selector. They are
  # checked per-crate here because the workspace-wide `cargo test --doc` cannot see this class of
  # breakage -- feature unification pulls `config` in via the umbrella crate, which is exactly how
  # four `impl_serde_type!` examples stayed broken without CI noticing.
  run_step "${doc_log}" \
    cargo test -p "${PACKAGE}" --doc "${feature_args[@]}" \
    || {
      test_st="FAIL"
      rung_failed=1
    }

  if [ -f "${test_log}" ]; then
    count="$(test_count "${test_log}")"
  fi
  if [ -f "${doc_log}" ]; then
    count=$((count + $(test_count "${doc_log}")))
  fi

  res_label+=("${label}")
  res_check+=("${check}")
  res_clippy+=("${clippy}")
  res_test+=("${test_st}")
  res_count+=("${count}")

  elapsed=$(($(date +%s) - start_all))
  # Mean of the configurations finished so far is the only honest ETA available.
  eta=$((elapsed * (TOTAL - n) / n))

  if [ "${rung_failed}" -eq 0 ]; then
    printf '[%d/%d] %-24s check ok   clippy ok     test ok   (%s tests) | elapsed %ds | eta ~%ds\n' \
      "${n}" "${TOTAL}" "${label}" "${count}" "${elapsed}" "${eta}"
  else
    FAILED=1
    printf '[%d/%d] %-24s check %-4s clippy %-6s test %-4s (%s tests) | elapsed %ds | eta ~%ds\n' \
      "${n}" "${TOTAL}" "${label}" "${check}" "${clippy}" "${test_st}" \
      "${count}" "${elapsed}" "${eta}"
    # The failing log, inline, so a CI run needs no artifact download. Only the
    # tail: a failing `cargo test` log is mostly the hundreds of tests that
    # passed, and the diagnosis is always at the end.
    for step in check clippy test; do
      case "${step}" in
        check) st="${check}"; log="${check_log}" ;;
        clippy) st="${clippy}"; log="${clippy_log}" ;;
        test) st="${test_st}"; log="${test_log}" ;;
      esac
      if [ "${st}" = "FAIL" ]; then
        echo "        --- cargo ${step} (${label}), last ${LOG_TAIL_LINES} lines ---"
        tail -n "${LOG_TAIL_LINES}" "${log}" | sed 's/^/        /'
        echo "        --- end cargo ${step} ---"
      fi
    done
  fi
done

echo
printf '%-24s  %-6s  %-6s  %-6s  %s\n' "configuration" "check" "clippy" "test" "tests"
printf '%-24s  %-6s  %-6s  %-6s  %s\n' \
  "------------------------" "------" "------" "------" "-----"
for i in "${!res_label[@]}"; do
  printf '%-24s  %-6s  %-6s  %-6s  %s\n' \
    "${res_label[${i}]}" "${res_check[${i}]}" "${res_clippy[${i}]}" \
    "${res_test[${i}]}" "${res_count[${i}]}"
done

echo
if [ "${FAILED}" -eq 0 ]; then
  echo "all ${TOTAL} configurations pass"
else
  echo "at least one configuration failed"
fi

exit "${FAILED}"
