<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/huggingface/tokenizers/actions"><img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg"></a>
    <a href="https://crates.io/crates/tokenizers"><img alt="Crates.io" src="https://img.shields.io/crates/v/tokenizers.svg"></a>
    <a href="https://docs.rs/tokenizers/"><img alt="Docs" src="https://docs.rs/tokenizers/badge.svg"></a>
    <a href="#footprint"><img alt="Size" src="https://img.shields.io/badge/gzipped-325%20KB-brightgreen"></a>
    <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue&cachedrop"></a>
</p>

The fastest tokenization library on all languages, all models, all hardwares. 

Our goal with `tokenizers` is to develop and maintain the industry's standard tokenization engine, making it the defacto place for everyone to contribute to the whole ecosystem.

### Release candidate: v1.0.0.rc.0

As we are switching from 0.23 to v1.0.0, the current library does not ship all features. If you are worried, check the [v1 features](#v1-features) section for more details of what we are bringing back.
We will publish a blog about what breaking changes we introduced. We strived to keep them as small as possible.

# Installation

```bash
pip install --pre tokenizers
```

# Usage

```python
>>> from tokenizers import Tokenizer
>>> tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B")   # or .from_file(path)
>>> tokenizer.encode("Hello, y'all! How are you 😁 ?")
Encoding(ids=[128000, 9906, 11, 379, 65948, 0, 2650, 527, 499, 27623, 223, 949], type_ids=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], attention_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
>>> tokenizer.tokenize("Hello")
['<|begin_of_text|>', 'Hello']
>>> tokenizer.decode([15339, 1917])
'hello world'
>>> tokenizer.decode_tokens(tokenizer.encode("Hello"))
['<|begin_of_text|>', 'Hello']
```

# Performances


To measure performances of `tokenizers`, we strongly advise to use `tokbench`.

## tokbench

**[tokbench](https://github.com/huggingface/tokbench)** is a standalone cross-engine benchmark we built because there was no honest way to compare most of the sota tokenization libraries.

It exists to remove the two ways tokenizer benchmarks usually mislead:

- **One timing loop, one process.** Every engine is measured by the same harness on the same bytes,
  rather than each project quoting its own number from its own rig.
- **An id-verification gate.** Every run is hashed and compared against the reference ids. An
  engine that computes *different* ids is marked `mismatch` and is never ranked. Being fast at the
  wrong answer is not a win, and this is where "supports every model" stops being a slogan and
  becomes a column: engines that decline non-Latin scripts, or quietly differ on them, show up as
  declined or mismatched cells instead of as speed.

Run it yourself:
[huggingface/tokbench](https://github.com/huggingface/tokbench).


# To come for v1

As we work toward v1, we are gonna bring back the entire python API that allows `transformers`' style of interacting with a tokenizer object:

## The target python API

```python
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.model = models.BPE.from_file("vocab.json", "merges.txt")
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.pre_tokenizer
```

## Training 

```python
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

# TODOs remaining before 1.0.0

- Improve `bitcannon`: we want to rewrite it a bit.
- Bring training back: this should be easy, but we want the perfs gains to come as well.
- Bring back offset output: this should be useful for people training.
- cpp / java / go bindings: there is a draft for C, we want to work on other bindings as well.
- Unroll more regex: we need to cover the most used ones, we might have missed one or two.
- `bitnorm`: more performance from optimized normalization, as this will bring breaking changes we want to make sure it gets to v1.

### After 1.0.0

- `tk-devices` — exploratory GPU encoding and batch decoding that keeps text and ids on the
  device: upload the vocabulary once, compute output positions in parallel, gather the bytes on the
  GPU. An optional component aimed at large batches, subject to prototyping and measurement.

<a name="footprint"></a>

# Crate details

## Crate size

```
make slim-size      # tk-encode + tk-serialize, minsize profile, stripped
                    # → 332799 bytes gzipped
```

Gzipped is the only honest number: Mach-O segments are 16 KiB-quantised, so the on-disk size of a
small binary is mostly padding. `make slimest` goes lower still (`minsize` plus a `std` rebuilt
without unwinding, nightly). The badge at the top is this number, measured on macOS — a stripped
ELF gzips to something slightly different.

## Sub-crates

<details>
<summary><b><code>tk-encode</code></b> — the inference half</summary>

The model engines (BPE, Unigram, WordPiece, WordLevel) and the full pipeline: `Normalizer`,
`PreTokenizer`, `Model`, `PostProcessor`, `Decoder`.

**Why separate:** inference is the only half that ships to production. Splitting it from training
means a serving binary never links a trainer, a corpus reader or a progress bar — which is most of
the 325 KB story.
</details>

<details>
<summary><b><code>bitcannon</code></b> — SIMD pre-tokenization</summary>

Unicode atom classification plus pre-tokenization as a **bitstream program** rather than a scalar
FSM. Follows *Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs*
(MICRO'25, [10.1145/3725843.3756052](https://doi.org/10.1145/3725843.3756052)): compile the grammar
into character-class bitstreams (one bit per input byte) plus boolean ops and carry-propagating
adds, so 64 input bytes are decided per 64-bit register op, branchlessly. The FSM's per-token
unpredictable branch disappears.

Ships byte-exact grammars for gpt2 / ByteLevel, cl100k, o200k, tekken, deepseek and kimi-k2.

**Why it is custom, and not a regex engine you could swap in.** The grammars never look at raw
bytes. They run on the tag stream `classify` emits — one `Atom` byte per codepoint, whose *low*
nibble is one of 16 coarse classes (`Letter`, `NumWord`, `Newline`, `Space`, `Mark`, `Punct`,
`Apostrophe`, …) and whose *high* nibble carries a refinement. That split is the whole trick: o200k
needs letter case, so `Letter` refines into `UpperLetter` / `LowerLetter`, while gpt2 — which does
not care — masks the refinement off for free (`& 0x0F` before a 16-entry SIMD LUT). One classifier
feeds every grammar, and a grammar pays only for the distinctions it actually asked for. A stock
regex engine has no such shared vocabulary to compile against.

**Zero Unicode dependencies at runtime.** Those tables are committed source, baked offline by
`bitmap_gen` (below) from `unicode-properties`. `bitcannon`'s entire runtime dependency list is
`ahash` — no `unicode-*` crate, no build script, so nothing that ships carries a Unicode table
crate or rebuilds one.

**Why separate:** it is a regex compiler, not a tokenizer — an independent artifact with its own
test surface, where every vectorised kernel is validated byte-for-byte against one scalar oracle.
Keeping it out of `tk-encode` is what makes "SIMD is pure speed, never correctness" checkable
rather than aspirational.
</details>

<details>
<summary><b><code>tk-serialize</code></b> — the reader</summary>

`from_json_file` turns a canonical `tokenizer.json` into a `PipelineTokenizer`. **No serde
anywhere.**

**Why separate:** serde's derive chain was a large, unconditional dependency sitting in front of
the one thing every user does exactly once. Reading a config is not the same job as encoding text,
and it should not tax it.
</details>

<details>
<summary><b><code>tk-convert</code></b> — the upgrade pass</summary>

`canonicalize_file` rewrites a `tokenizer.json` written by any older version of this library into
the canonical form the reader accepts. A pure JSON→JSON rewrite; it depends on nothing but
`std::path` and `serde_json`.

**Why separate:** every config ever published stays readable without the runtime carrying a decade
of compatibility branches. `cargo tree -p tk-convert -e normal` is 8 nodes.
</details>

<details>
<summary><b><code>tk-train</code></b> — the training half</summary>

The `Trainer` trait, every concrete `*Trainer`, `TrainerWrapper` and the `Trainable` extension.

**Why separate:** training is a batch job on a workstation; inference is a hot loop in a server.
They have opposite constraints, so they get opposite dependency budgets.
</details>

<details>
<summary><b><code>bitmap_gen</code></b> — dev-only table generator</summary>

`cargo run -p bitmap_gen` regenerates `bitcannon`'s committed classify tables from
`unicode-properties`, emitting one `Atom` tag per codepoint.

**Why separate — this is what buys the zero Unicode dependency.** `unicode-properties` is a
dependency of *this* crate and of nothing else: the tables it produces are checked into
`bitcannon/src/classify/atom_tables.rs` as ordinary source, so the Unicode data is resolved once, at
development time, by a crate that is never linked into anything that ships and is never published.
No build script either — `bitcannon` compiles with no code generation step, and a release binary
contains the tags without containing a Unicode crate to derive them. The release workflow re-runs
the generator and fails if the committed table differs, so "baked" cannot silently mean "stale".
</details>

<details>
<summary><b><code>tokenizers</code></b> — umbrella</summary>

A thin re-export so existing `tokenizers::…` paths keep working. Depend on this one unless you know
you want less.
</details>

## Bindings

| | status |
|---|---|
| [Rust](tokenizers) | ✅ reference implementation |
| [Python](bindings/python) | ✅ |
| [Node.js](bindings/node) | ✅ |
| C / C++ | 🚧 planned |
| [Ruby](https://github.com/ankane/tokenizers-ruby) | community, external repo |

## Hardware

**No SIMD is required.** Every kernel has a portable path that is always compiled and is the
byte-exact test oracle for its vectorised siblings, so correctness never depends on a kernel being
present — only throughput does.

<details>
<summary>Which ops are hardware-adapted</summary>

| op | aarch64 | x86_64 | wasm32 | portable fallback |
|---|---|---|---|---|
| Unicode atom classify | NEON (baseline) | AVX-512 VBMI → SSE4.1/SSSE3, runtime-detected | SIMD128 | `classify_scalar` |
| bitstream block build | NEON | SSE/AVX | — | `build_block_scalar` |
| literal & added-token scan | NEON | x86_64 | — | scalar |
| vocab bucket nibble match | NEON | — | — | scalar |

aarch64 selects at compile time (NEON is baseline). x86_64 dispatches at runtime via
`is_x86_feature_detected!`, so one binary covers every x86_64 CPU since 2008. wasm32 needs the
`simd128` target feature; without it, the scalar walk.
</details>

## Getting it

```bash
pip install --pre tokenizers
```

```toml
[dependencies]
tokenizers = "1.0.0-rc.0"
```

# Acknowledgements
This work stands on the shoulders of a lot of open source projects. Thanks to all of them for their great work.

[gigatoken](https://github.com/marcelroed/gigatoken),
[tiktoken](https://crates.io/crates/tiktoken-rs), [kitoken](https://crates.io/crates/kitoken),
[tokie](https://crates.io/crates/tokie), [fastokens](https://crates.io/crates/fastokens),
[wordchipper](https://crates.io/crates/wordchipper) and
[ai-tokenizer](https://www.npmjs.com/package/ai-tokenizer) each pushed on what a fast tokenizer can
be; we read that work, and several of the ideas above reached us because another project showed
they were worth trying.

Docs: [guide](https://huggingface.co/docs/tokenizers/index) ·
[quicktour](https://huggingface.co/docs/tokenizers/quicktour) ·
[docs.rs](https://docs.rs/tokenizers/)
