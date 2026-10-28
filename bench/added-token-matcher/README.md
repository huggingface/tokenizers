# Added-token matcher: buckets vs aho-corasick

Measures main's bucket matcher (`tk-encode/src/vocab/buckets.rs` at `d9a5e8af`) against this branch, which
replaces it with the `aho-corasick` crate (`AddedTokenMatcher`, `MatchKind::LeftmostLongest`). Both arms are the
same harness built twice: `--features buckets` against main, and without the feature against main plus this
worktree's diff (`branch.diff`, embedded in the job so nothing is pushed).

| what | how |
|---|---|
| matcher alone | each model's added tokens (content and id from `tokenizer.json`) in one matcher; every match found, restarting after each match like `extract_next`. Whole 20 MB text in one call, and 4 KB pieces |
| end to end | `from_json` of the model's canonicalized `tokenizer.json`, `encode` with `EncodeOptions::no_specials()` on 4 KB pieces, one thread |
| memory | counting global allocator: bytes left allocated by the matcher build and peak during it; bytes left allocated by the whole `from_json` |
| gate | a hash of every `(start, end, id)` match and of every encoded id, which must agree between arms |

Corpora (`corpora.py`, revisions printed in the log), 20 MB each:

- **english**: FineWeb `sample/10BT/000_00000.parquet`.
- **code**: the `patch` column of SWE-smith trajectories, deduplicated (unified diffs of Python repositories).
- **agentic**: SWE-smith `tool` trajectories rendered through each model's own chat template, so the text holds
  that model's real markers. A model without a template (gpt2) gets `role: content` lines.

Models, few added tokens to many: gpt2, gpt-oss-20b, Qwen3-8B, DeepSeek-V3, Nemotron-3-Nano, surya-ocr-2 (40
distinct first bytes), Gemma 3 (`gaunernst/gemma-3-27b-it-int4-awq`, an ungated copy of its tokenizer).

## Run

> [!WARNING]
> `make job` launches a Hugging Face Job on your account: `cpu-performance`, billed per minute (about 30 min).

```sh
make job                  # bundle + one job; `make jobs` for two hosts
make job ROUNDS=1 SECS=0.05 FLAVOR=cpu-upgrade   # quick pilot
make fetch ID=<job-id>    # results/<job-id>/summary.md
```

`bundle.py` writes `start_job.sh` from `start_job.in.sh`, `harness/` and `corpora.py`, with `git diff d9a5e8af --
tokenizers` of this worktree (new files need `git add -N`). Settings: `MAIN`, `ROUNDS` (default 5, arms
alternate order each round), `SECS` (seconds per timed cell, default 1).

Locally: copy `harness/` and point the two `/w/branch` paths in `Cargo.toml` at a checkout; add
`--features buckets` for a checkout that still has `buckets.rs`.
