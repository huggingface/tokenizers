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

The SOTA tokenization library on all languages, all models, all hardwares. 

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


<img width="1238" height="677" alt="image" src="https://github.com/user-attachments/assets/49c672aa-f7c3-4eb5-a89c-99703f8ee38f" />
For more details, checkout https://huggingface-tokenizers-v1.static.hf.space/index.html.

# Bindings

| | status |
|---|---|
| [Rust](tokenizers) | ✅ reference implementation |
| [Python](bindings/python) | ✅ |
| [Node.js](bindings/node) | ✅ |
| C / C++ / Java / Go/ etc | 🚧 planned |
| [Ruby](https://github.com/ankane/tokenizers-ruby) | community, external repo |


# v1.0.0 roadmap:

As we work toward v1, we are gonna bring back the entire python API that allows `transformers`' style of interacting with a tokenizer object:

## TODOS:

- Improve `bitcannon`: we want to rewrite it a bit.
- Bring training back: this should be easy, but we want the perfs gains to come as well.
- Bring back offset output: this should be useful for people training.
- cpp / java / go bindings: there is a draft for C, we want to work on other bindings as well.
- Unroll more regex: we need to cover the most used ones, we might have missed one or two.
- `bitnorm`: more performance from optimized normalization, as this will bring breaking changes we want to make sure it gets to v1.

## After 1.0.0

- `tk-devices` — exploratory GPU encoding and batch decoding that keeps text and ids on the
  device: upload the vocabulary once, compute output positions in parallel, gather the bytes on the
  GPU. An optional component aimed at large batches, subject to prototyping and measurement.

<a name="footprint"></a>

# Crate details

## Crate size
If you want the minimal crate size, we recommend you to use this:
```
make slim-size      # tk-encode + tk-serialize, minsize profile, stripped
                    # → 332799 bytes gzipped
```

## Sub-crates

<details>
<summary><b><code>tk-encode</code></b> — the inference engine</summary>

The models (BPE, Unigram, WordPiece, WordLevel) and the full pipeline: `Normalizer`,
`PreTokenizer`, `Model`, `PostProcessor`, `Decoder`.

**Why separate:** you don't need all features to serve a model in production settings. Splitting it from training
means a serving binary never links a trainer, a corpus reader, legacy converters, etc.
</details>

<details>
<summary><b><code>bitcannon</code></b> — bitstream pre-tokenization</summary>

Unicode tag classification plus pre-tokenization as a **bitstream program**. Follows *Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs*
(MICRO'25, [10.1145/3725843.3756052](https://doi.org/10.1145/3725843.3756052)) but we added our own algorithm to make sure we don't need a parabix engine. 
Characters are first classified into "tags", which use a sparse table generated when compiling `bitcannon`. This means there are 0 depencencies to the heavy unicode tables.

Ships byte-exact grammars for gpt2 / ByteLevel, cl100k, o200k, tekken, deepseek and kimi-k2.
The classes or tags each have a high and low nibble (encoded on a u8), and carry refinement when needed. Using u8 tags allows us to leverage less bitsream, meaning have a simpler / faster pre tokenization.

**Why separate:** we just thought it might be useful to some people someday. As far as we know, this implementation is the fastest for every single pretokenizer regex on CPU.
</details>

<details>
<summary><b><code>tk-serialize</code></b> — the reader</summary>

`from_json_file` turns a canonical `tokenizer.json` into a `PipelineTokenizer`.

**Why separate:** this just makes it optional for some usecases!
</details>

<details>
<summary><b><code>tk-convert</code></b> — the upgrade pass</summary>

`canonicalize_file` rewrites legacy a `tokenizer.json` written by any older version of this library into
the canonical form the reader accepts. A pure JSON→JSON rewrite; it depends on nothing but
`std::path` and `serde_json`.

**Why separate:** every config ever published stays readable without the runtime carrying a decade
of compatibility branches. But, the legacy code amounted to quite a lot of the final crate size. Disabling it allows on device builds to be smaller.
</details>

<details>
<summary><b><code>tk-train</code></b> — the training half</summary>

The `Trainer` trait, every concrete `*Trainer`, `TrainerWrapper` and the `Trainable` extension.

**Why separate:** training is a batch job on a workstation; inference is a hot loop in a server.
They have opposite constraints, so they get opposite dependency budgets.
</details>

<details>
<summary><b><code>bitmap_gen</code></b> — dev-only table generator</summary>

`cargo run -p bitmap_gen` regenerates `bitcannon`'s committed classify tables from `unicode-properties`, emitting one `Atom` tag per codepoint.

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
