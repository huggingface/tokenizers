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
<br>

The fastest tokenizer library on the world's text, and the only fast one that runs **every** model
— BPE, Unigram, WordPiece and WordLevel, from one `tokenizer.json`. Every other engine near the top
of the table buys its speed by supporting less.

- **881 MB/s** single thread and **6.5 GB/s** at 8 threads (94% scaling efficiency), measured
  across the whole script matrix and not an English subset.
- **5–41×** faster than `tokenizers` 0.23.1 on every model and every corpus.
- **325 KB** gzipped, no hardware requirement — SIMD is pure speed, never correctness.

## Performance

All numbers from [**tokbench**](https://github.com/huggingface/tokbench): one shared timing loop,
one process, and an FNV-1a id-verification gate — an engine that computes different ids is marked
`mismatch` and never ranked. Being fast at the wrong answer is not a win.

### The global set

[![head to head](https://raw.githubusercontent.com/huggingface/tokbench/tokbench-init/figs/03-headtohead.png)](https://github.com/huggingface/tokbench)

Head-to-head against gigatoken, the closest competitor. One process, median of 7, idle machine,
single thread, over all 20 script cells — not a Latin subset:

| | tokenizers | gigatoken |
|---|---:|---:|
| median, all scripts | **881 MB/s** | 590 MB/s |
| cells won | **15 / 20** | 5 / 20 |
| gpt2 | **698** | 554 |
| llama-3 | **947** | 612 |
| cells it will even attempt | **203 / 210** | 145 (declines 65) |

We win non-Latin decisively — chinese **1.90×**, korean 1.75×, arabic 1.67×, greek 1.60×, russian
1.55×, thai 1.53× — and threading does not rescue the gap, because both engines scale at ~94%: on
Chinese it is still 6826 vs 3657 MB/s at 8 threads.

<sub>Figures from the <code>pipeline #2279 + #2296</code> snapshot. <code>bitsplit</code>, the thread
pool and the load-time fold have all landed since; this table has not been re-run.</sub>

### Where we lose, plainly

Two places, and both are in tokbench for anyone to check:

- **Latin and dense text.** gigatoken is faster there — english 0.87×, code 0.82×.
- **tokbench's 10-engine ranking.** Over the 23 cells all ten engines verify, gigatoken leads on
  median throughput, **238.6 vs our 170.1 MB/s**. That subset is Latin-heavy by construction — it
  is the intersection of what ten engines support, and engines that decline non-Latin shrink it
  toward English. It is a real number about a narrow set, which is why the global set is quoted
  above rather than instead of it.

The coverage column cuts both ways too: gigatoken is byte-exact on all 145 cells it accepts and
simply declines the other 65. We attempt 203 of 210 and are exact on 174 — see tokbench's coverage
table for the rest.

[![overall throughput](https://raw.githubusercontent.com/huggingface/tokbench/tokbench-init/figs/01-overall.png)](https://github.com/huggingface/tokbench)

Every engine ranked over the cells it got byte-exact. Log scale — the spread is three orders of
magnitude. Full matrix, caveats and the interactive dashboard:
[huggingface/tokbench](https://github.com/huggingface/tokbench).

<a name="footprint"></a>
### Footprint

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

## Quick start

```bash
pip install tokenizers
```

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
```

```toml
[dependencies]
tokenizers = "1.0.0-rc.0"
```

Docs: [guide](https://huggingface.co/docs/tokenizers/index) ·
[quicktour](https://huggingface.co/docs/tokenizers/quicktour) ·
[docs.rs](https://docs.rs/tokenizers/)
