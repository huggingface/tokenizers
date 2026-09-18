# Required for v1

Things rc0 deliberately drops. Each is a decision, not an oversight, and each has to come back
before 1.0 — as pipeline-native code, not by resurrecting the config layer.

**Context.** rc0 strips `tk-convert` down to `convert.rs` alone: the legacy `tokenizer.json` →
canonical `tokenizer.json` upgrade, 1,207 lines whose only dependencies are `std::path`,
`serde_json` and a `thiserror` derive on `ConvertError`. Everything else in that crate —
`Tokenizer`, `TokenizerImpl`, `TokenizerBuilder`, the five wrapper enums, the config models,
`added_vocabulary.rs`, `lowering.rs` — is deleted, 9,369 lines of the crate's 12,979. The runtime is
`tk_encode::pipeline::PipelineTokenizer`; the reader is `tk-serialize`.

The crate's dependency graph goes with it: `tk-convert` no longer depends on `tk-encode`, nor on
`serde`, `ahash`, `daachorse`, `dary_heap`, `rand`, `regex`, `log` or `paste`. `cargo tree
-p tk-convert -e normal` is 8 nodes, of which 6 are `thiserror`'s proc-macro chain.

Test count across `cargo test --workspace`: **544 → 373 passing**, 32 → 19 suites. Where the 171 went:

| where | lost | how |
|---|---:|---|
| `tk-convert` unit tests | 65 | deleted with `src/**` (89 → 24, the 24 being `convert.rs`'s own) |
| `tk-convert` doc tests | 7 | deleted with the module docs |
| `tk-convert` integration tests | 54 | §9: `lowering` 26, `lowering_pre_tokenizers` 18, `bpe_pipeline_oracle` 6, `slim_vs_config` 5 → 1 |
| `tokenizers` umbrella tests | 33 | §1: out of the build, source kept |
| `tokenizers` umbrella doc test | 1 | its example authored a `BPE` |
| `tk-train` unit + doc | 11 | §2: out of the workspace, source kept |

§9 lists what was deleted outright; §1 and §2 list what is out of the build but still in the tree.

---

## 1. An authoring surface (the setters)

**What breaks:** the Python bindings. `PipelineTokenizer` has **0** setters — it is read-only. The
bindings hold config wrapper enums in **195** places:

| type | references in `bindings/` |
|---|---:|
| `NormalizerWrapper` | 53 |
| `PreTokenizerWrapper` | 51 |
| `DecoderWrapper` | 45 |
| `PostProcessorWrapper` | 24 |
| `ModelWrapper` | 15 |
| `TokenizerImpl` | 7 |

Python does `tokenizer.normalizer = Lowercase()`, and `PyNormalizer` &co. wrap
`Arc<RwLock<…Wrapper>>` — which is also what they pickle through, so the wire format of
`pickle.dumps(tokenizers.models.BPE(...))` is part of this.

**What v1 needs:** a mutable pipeline builder that the bindings' setters drive, so `PyTokenizer`
stops holding wrapper enums. Getters exist already; the gap is entirely on the write side.

**Also gone, and not just the bindings:** the umbrella crate's *own* test and bench suites. Every one
of them authors a tokenizer in process — `Tokenizer::new(BPE::default())`, `with_pre_tokenizer`,
`add_tokens`, `save`, the trainers — so none of them can be repointed at a read-only pipeline. That
was all 9 benches (`tokenizers/benches/*`, 1,337 lines) and all 8 test files (`tokenizers/tests/*`,
1,464 lines, **33 passing tests**: `added_tokens` 5, `documentation` 6, `offsets` 5, `serialization`
11, `stream` 2, `training` 2, `unigram` 2, `from_pretrained` 0).

Both are now **out of the tree**: `tokenizers/tests/` went with the strip itself, and
`tokenizers/benches/` with the bench sweep. The umbrella package still carries `autotests = false` /
`autobenches = false`, which is why `cargo build --workspace --all-targets` stayed green in between.

For the **tests** that is a debt this section owes: they are the written specification for the
builder it asks for, so recover them from git history and bring them back *as the acceptance test*
for it rather than rewriting from scratch. Nothing else in the workspace covers
`with_padding`/`with_truncation` behaviour, `save` round-trips, or `decode_stream` over an authored
tokenizer, so until they run again those are untested.

For the **benches** it is not: they are not coming back in this shape. The cross-engine benchmarks
live in [tokbench](https://github.com/huggingface/tokbench), and what stays in-tree is two criterion
files against the pipeline (`tk-serialize/benches/{encode,decode}.rs`) — the only crate that can
build a `PipelineTokenizer`. Nothing about them waits on the builder.

## 2. Trainers

**What breaks:** `tk-train` does not compile. It builds *config* models, importing from `tk_convert`
across 8 files:

    tk_convert::TokenizerImpl
    tk_convert::ModelWrapper
    tk_convert::AddedToken
    tk_convert::models::bpe::{BPE, WithFirstLastIterator, Word}
    tk_convert::models::wordpiece::from_bpe

**What v1 needs:** trainers that emit pipeline models directly. Until then training is unavailable.
This was an explicit rc0 call — inference first.

**How it is wired now:** `tk-train` is out of the workspace — `exclude = ["tk-train"]` in
`tokenizers/Cargo.toml`, not merely absent from `members`, and the umbrella's optional `tk-train`
dependency is gone with it (the default-on `train` feature would otherwise have kept building it).
The crate's source is untouched in the tree. Three umbrella features went with it: `train`,
`esaxx_fast` and `parity-aware-bpe`, so `tokenizers`'s default set is now `["progressbar", "onig"]`.
The `tokenizers::models::*::trainer` legacy module paths and `TokenizerTrainExt` no longer resolve.
CI's readme loop still runs `cargo readme --project-root tk-train`, which works on an excluded crate
and still matches `tk-train/README.md` byte for byte — verified, not assumed. `tk-train`'s own 10
unit tests and 1 doc test stop running.

## 3. Writing a `tokenizer.json`

**What breaks:** `Tokenizer::save` was the only way to emit one.

**Status:** being replaced by `tk_serialize::to_json`, behind the already-declared `serialize`
feature (off by default). Serde-free, and it must emit floats that parse back through
`tk-serialize`'s own parser to *identical bits* — that parser deliberately reproduces `serde_json`'s
non-correctly-rounded arithmetic, so shortest-form output is not safe here.

## 4. Loading a model from its own files

Deleted with the config models. These read `vocab.json` + `merges.txt`, `vocab.txt`, and bare
Unigram files rather than a `tokenizer.json`:

    BPE::from_file, models::wordpiece::{from_file, from_bytes}, models::wordlevel::{from_file, read_file},
    models::unigram::{load, save}

`tk_encode::models::unigram::Model::save` currently errors with a message pointing at
`tk_convert::ModelWrapper::save`; that message needs rewording once the target is gone.

## 5. Hub and file constructors on the config object

`Tokenizer::{from_file, from_bytes, from_pretrained}` (63 public functions in
`tk-convert/src/tokenizer/mod.rs` in total). `tk_serialize::{from_json, from_json_file}` covers the
read path; `from_pretrained` needs a pipeline-native equivalent — `tk-encode` already has
`utils/from_pretrained.rs` behind the `http` feature to build on.

## 6. Truncation and padding on the object model

`with_truncation` / `with_padding` and the `TruncationParams` / `PaddingParams` config types. The
pipeline currently *swallows* truncation and padding declared in a `tokenizer.json` rather than
erroring, which is the more dangerous of the two behaviours — a config that asks for truncation gets
silently untruncated output. Worth fixing before v1 even if the feature itself lands later.

## 7. The cross-reader id oracle

**What breaks:** 4 of the 5 tests in `tk-convert/tests/slim_vs_config.rs` use `tk_convert::Tokenizer`
as the reference the slim reader is checked against — the byte-level fold check, the batched-path
check, `decode_matches_the_config_path_on_the_real_configs`, and
`encode_matches_the_config_path_on_every_real_config` (266 comparisons over 19 configs). All of them
die with the config layer.

**What survives:** `canonicalizing_first_does_not_move_ids` (152 comparisons, 4 Unigram configs) —
it only needs `canonicalize_str` and `from_json`. Plus, in `tk-serialize`,
`agrees_with_serde_json_on_every_real_config`, which compares whole parse trees against `serde_json`
with bit-exact floats over all 22 configs, and the reader/writer round-trip gate.

**Honest accounting:** tree-equality proves the *parser*; the round-trip proves reader-against-writer.
Neither proves the reader against an *independent implementation*, which is what the deleted oracle
did.

**Correction: that replacement already existed, and it survived.** `tk-convert/tests/pipeline_oracle.rs`
(9 tests, 9 configs) and `tk-convert/tests/pipeline_decode_oracle.rs` (7) already check the pipeline's
ids and decoded text against `tokenizers_release::Tokenizer` — the released crate, an independent
implementation — and say so in their own headers: *"not the in-tree legacy `Tokenizer`, which is being
removed (oracles must not depend on the thing being removed)"*. They used
`tk_convert::Tokenizer::from_file` only as the pipeline *builder*, so both were repointed at
`tk_serialize::from_json_file` and kept. This section's "worth doing before v1" is therefore already
done; what is missing is only that both files sit behind the non-default `bench-baseline` feature, so
`cargo test --workspace` does not run them:

    cargo test -p tk-convert --features bench-baseline --test pipeline_oracle
    cargo test -p tk-convert --features bench-baseline --test pipeline_decode_oracle

Promoting them into the default gate needs the released crate to become a non-optional dev-dependency,
and `pipeline_decode_oracle`'s `bert_wiki` is a known deliberate failure (`PipelineWordPiece` has no
id→token direction), so that has to land first.

**One decode-oracle test did die:** `non_special_added_token_survives_skip`, which added the same
non-special token to both implementations via `add_tokens` before comparing `skip_special_tokens`
behaviour. There is no way to add a token to a read-only pipeline, so it goes with §1 and comes back
with the setters. Nothing else now covers the special-vs-non-special distinction in `decode`.

**And one benchmark lost its subject:** `fixture_bench` used to inject its own added tokens
(`<|xs0|>`-style specials plus word-shaped markers) into every model it loaded, which is what made the
`added_special_sparse` and `added_normalized_sparse` fixtures added-token workloads. That went through
`Tokenizer::{add_special_tokens, add_tokens}`. It is removed from *both* sides — leaving it on the
released baseline alone would have broken the `ids_match` gate for every model — so those two fixtures
still run and still compare fairly, but their markers are now ordinary text and the added-token scan
path is unmeasured. Restoring it needs either the setters or a config rewrite that mints ids for the
markers before either implementation loads the file.

## 8. The umbrella re-export surface

`tokenizers/src/lib.rs` re-exports 17 groups from `tk_convert` so that `tokenizers::{Tokenizer,
models::bpe::BPE, normalizers::NormalizerWrapper, …}` resolve at their historical paths. Those paths
disappear. Both bindings and any downstream Rust user of the `tokenizers` facade are affected.

**What the facade is now:** the `tk_encode` modules (`decoders`, `models`, `normalizers`,
`pipeline`, `pre_tokenizers`, `processors`, `tokenizer`, `utils`, `vocab`) plus the four
`canonicalize_*` items and `ConvertError` from `tk_convert`. The per-module merges are gone, so every
`tokenizers::<module>::Sequence`, every `*Wrapper`, `Tokenizer`, `TokenizerImpl`, `TokenizerBuilder`,
`AddedToken`, the config-shaped `BPE`, `models::wordpiece::{from_file, from_bpe, …}`,
`models::wordlevel::{from_file, read_file}` and `models::unigram::load` are unresolvable. The
umbrella keeps a `tk-convert` dependency for exactly one reason: to keep `tokenizers::canonicalize_file`
reachable from the facade.

## 9. Deleted outright (not deferred behind a flag)

Everything above is dropped but recoverable from code still in the tree. These are gone:

| what | lines | why it cannot come back as-is |
|---|---:|---|
| `tk-convert/tests/lowering.rs` | 707 | 26 tests of `lowering.rs`, the config→pipeline lowering. There is no config to lower. |
| `tk-convert/tests/lowering_pre_tokenizers.rs` | 584 | 18 tests of `to_normalizer_and_split` over `PreTokenizerWrapper`. |
| `tk-convert/tests/bpe_pipeline_oracle.rs` | 221 | 6 tests whose *oracle* was the legacy engine — "the legacy engine is the oracle: it is the code main runs". The oracle itself is deleted, so this is not repointable the way §7's two are. Its coverage (10 corpora × 4 BPE models, incl. llama-2's non-byte-level `Atoms::Chars` path and the unsafe-to-batch merges the `SAFE` flag exists for) is the largest single verification loss of the strip, and `pipeline_oracle.rs` does not replace it: that one runs 9 configs over fixture windows, not 400 kB per corpus. |
| `tk-convert/benches/bpe_model_benchmark.rs` | 321 | Benched `ModelWrapper` and the config-shaped `BPE` against `PipelineModel`. Both operands are half gone. |

`tk-convert/src/**` beyond `convert.rs` — 63 unit tests across the wrapper enums, the config `BPE`
(`models/bpe/tests.rs` alone had 25, including the `legacy_parity` set pinning the two BPE engines
against each other), `added_vocabulary.rs` (12) and the `Tokenizer` itself (6+2) — went with the code
they test. `tk-convert`'s 7 doc tests went with the module docs.
