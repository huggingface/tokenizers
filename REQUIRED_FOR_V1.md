# Required for v1

Things rc0 deliberately drops. Each is a decision, not an oversight, and each has to come back
before 1.0 — as pipeline-native code, not by resurrecting the config layer.

**Context.** rc0 strips `tk-convert` down to `convert.rs` alone: the legacy `tokenizer.json` →
canonical `tokenizer.json` upgrade, 1,207 lines whose only dependencies are `std::path` and
`serde_json`. Everything else in that crate — `Tokenizer`, `TokenizerImpl`, `TokenizerBuilder`, the
five wrapper enums, the config models, `added_vocabulary.rs`, `lowering.rs` — is deleted, about
7,500 lines. The runtime is `tk_encode::pipeline::PipelineTokenizer`; the reader is `tk-serialize`.

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
did. The nearest replacement is the released crate, already available as `tokenizers-release` behind
`tk-convert`'s `bench-baseline` feature and used by `fixture_bench` for exactly that purpose. Wiring
it as an id oracle is the cheapest way to get the guarantee back and is worth doing before v1.

## 8. The umbrella re-export surface

`tokenizers/src/lib.rs` re-exports 17 groups from `tk_convert` so that `tokenizers::{Tokenizer,
models::bpe::BPE, normalizers::NormalizerWrapper, …}` resolve at their historical paths. Those paths
disappear. Both bindings and any downstream Rust user of the `tokenizers` facade are affected.
