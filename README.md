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

The fastest tokenizer library on all language, all models, all hardwares.

**v1' release candidate.** `1.0.0-rc.0` is on crates.io and PyPI, and the associated blog post can be found here: **[tokenizers v1](https://huggingface-tokenizers-v1.static.hf.space/index.html)**.

<p align="center">
  <a href="https://huggingface-tokenizers-v1.static.hf.space/index.html">
    <img src="assets/single-thread-throughput.svg" width="700"
         alt="Single-thread tokenization throughput, median MB/s: tokenizers v1 131.2, gigatoken 128.7, fastokens 60.6, wordchipper 50.7, tiktoken 28.8, kitoken 26.8, tokie 25.9, tokenizers 0.23.1 8.7.">
  </a>
</p>

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

## What the rc actually gives you today

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B")   # or .from_file(path)

tokenizer.encode("Hello, y'all! How are you 😁 ?")   # -> Encoding
tokenizer.encode_batch([...])
tokenizer.tokenize("Hello")                          # -> ["Hello"]
tokenizer.decode([15339, 1917])
tokenizer.decode_tokens(encoding)
tokenizer.padding = None                             # get/set padding
```

### Remaining before 1.0.0

- Improve `bitsplit`
- Bring training back
- Apply performance improvement to trainer
- Bring back offset output
- cpp / java / go bindings
- GPU encode / decode
- Unroll more regex
- `bitnorm`: more performance from optimized normalization

### After 1.0.0

- `tk-devices` — exploratory GPU encoding and batch decoding that keeps text and ids on the
  device: upload the vocabulary once, compute output positions in parallel, gather the bytes on the
  GPU. An optional component aimed at large batches, subject to prototyping and measurement.

## Performance

Single thread, the closest competitor is [gigatoken](https://github.com/marcelroed/gigatoken), kudos to the authors!
In all fairness, they are still faster when the cache is unbound, but on the same cache size, tokenizers performs better!
We are gonna work a bit on unbound cache performances to make sure we leverage their ideas. 

### Against 0.23.1, by model and by language

### Latency and decode

## tokbench

Every number above comes from **[tokbench](https://github.com/huggingface/tokbench)**, a standalone cross-engine benchmark we built because there was no honest way to compare these
libraries.

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

<a name="footprint"></a>
## Rust crate size

```
make slim-size      # tk-encode + tk-serialize, minsize profile, stripped
                    # → 332799 bytes gzipped
```

Gzipped is the only honest number: Mach-O segments are 16 KiB-quantised, so the on-disk size of a
small binary is mostly padding. `make slimest` goes lower still (`minsize` plus a `std` rebuilt
without unwinding, nightly). The badge at the top is this number, measured on macOS — a stripped
ELF gzips to something slightly different.

## Crates

<details>
<summary><b><code>tk-encode</code></b> — the inference half</summary>

The model engines (BPE, Unigram, WordPiece, WordLevel) and the full pipeline: `Normalizer`,
`PreTokenizer`, `Model`, `PostProcessor`, `Decoder`.

**Why separate:** inference is the only half that ships to production. Splitting it from training
means a serving binary never links a trainer, a corpus reader or a progress bar — which is most of
the 325 KB story.
</details>

<details>
<summary><b><code>bitsplit</code></b> — SIMD pre-tokenization</summary>

Unicode atom classification plus pre-tokenization as a **bitstream program** rather than a scalar
FSM. Follows *Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs*
(MICRO'25, [10.1145/3725843.3756052](https://doi.org/10.1145/3725843.3756052)): compile the grammar
into character-class bitstreams (one bit per input byte) plus boolean ops and carry-propagating
adds, so 64 input bytes are decided per 64-bit register op, branchlessly. The FSM's per-token
unpredictable branch disappears.

Ships byte-exact grammars for gpt2 / ByteLevel, cl100k, o200k, tekken, deepseek and kimi-k2.

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

`cargo run -p bitmap_gen` regenerates `bitsplit`'s committed classify tables from
`unicode-properties`.

**Why separate:** the Unicode tables are baked and committed, so the generator is never linked into
anything that ships. Not published.
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

This work stands on a lot of open source. [gigatoken](https://github.com/marcelroed/gigatoken),
[tiktoken](https://crates.io/crates/tiktoken-rs), [kitoken](https://crates.io/crates/kitoken),
[tokie](https://crates.io/crates/tokie), [fastokens](https://crates.io/crates/fastokens),
[wordchipper](https://crates.io/crates/wordchipper) and
[ai-tokenizer](https://www.npmjs.com/package/ai-tokenizer) each pushed on what a fast tokenizer can
be; we read that work, and several of the ideas above reached us because another project showed
they were worth trying.

Docs: [guide](https://huggingface.co/docs/tokenizers/index) ·
[quicktour](https://huggingface.co/docs/tokenizers/quicktour) ·
[docs.rs](https://docs.rs/tokenizers/)
