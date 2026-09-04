# tk-encode

The inference half of `tokenizers`, written in Rust.
Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

## What is a Tokenizer

A tokenizer works as a pipeline, it processes some raw text as input and outputs an `Encoding`.
The various steps of the pipeline are:

1. The `Normalizer`: in charge of normalizing the text. Common examples of normalization are
   the [unicode normalization standards](https://unicode.org/reports/tr15/#Norm_Forms), such as `NFD` or `NFKC`.
   More details about how to use the `Normalizers` are available on the
   [Hugging Face blog](https://huggingface.co/docs/tokenizers/components#normalizers)
2. The `PreTokenizer`: in charge of creating initial words splits in the text. The most common way of
   splitting text is simply on whitespace.
3. The `Model`: in charge of doing the actual tokenization. An example of a `Model` would be
   `BPE` or `WordPiece`.
4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything relevant
   that, for example, a language model would need, such as special tokens.

### Loading a tokenizer and encoding

This crate is the runtime only: it knows how to *encode*, not what a `tokenizer.json` looks
like. Reading one is `tk-serialize`'s job — `tk_serialize::from_json_file` returns the
[`pipeline::PipelineTokenizer`] built here, parsed with `hifijson`, a JSON lexer with no
required dependencies, rather than `serde_json`. (The example lives there, because only that
crate can compile it.)

Two separate reasons, and they are worth keeping apart. `serde_json` is out on *weight*: it
pulls `serde`, `serde_core`, `itoa` and `memchr`, and the whole point of splitting these crates
is that an inference build links no serde at all. `hifijson` has no dependencies whatsoever.

`hifijson` rather than any other small parser is out of *arithmetic*. A Unigram's `vocab` scores
decide the Viterbi lattice, and `serde_json`'s rounding is already baked into ids that ship
today — so a parser that converts numbers to floats its own way moves them, and being more
accurate is a bug, not a fix. `hifijson` never parses a float at all: its `Num` is the raw
`&str`, which is what lets `tk-serialize` reassemble the `f64` to match `serde_json` bit for bit.

There is no `Tokenizer` type here to build up and mutate. What you get is a finished
[`pipeline::PipelineTokenizer`] — from `tk_serialize::from_json_file`, or from
[`pipeline::PipelineTokenizer::from_parts`] if you assemble the components yourself — and once
built it is read-only: getters, `encode` and `decode`, but no setters, no `save`, and no
`from_pretrained` constructor. Upgrading a config written by an older version is `tk-convert`'s
one remaining job (`canonicalize_file`, re-exported by the `tokenizers` umbrella crate). What an
authoring surface would have to add back is `REQUIRED_FOR_V1.md` at the repository root.

Training is not part of the shipped crate either. `tk-train` is in the tree and builds on its
own, but it is `exclude`d from the workspace and nothing in `tokenizers` depends on it.

## Additional information

- tokenizers is designed to leverage CPU parallelism when possible. The level of parallelism is determined
  by the total number of core/threads your CPU provides but this can be tuned by setting the `RAYON_RS_NUM_THREADS`
  environment variable. As an example setting `RAYON_RS_NUM_THREADS=4` will allocate a maximum of 4 threads.
  **_Please note this behavior may evolve in the future_**

## Features

- **progressbar**: The progress bar visualization is enabled by default. It might be disabled if
  compilation for certain targets is not supported by the [termios](https://crates.io/crates/termios)
  dependency of the [indicatif](https://crates.io/crates/indicatif) progress bar.

- **http**: This feature enables downloading the tokenizer via HTTP. It is disabled by default.
  It compiles `utils::from_pretrained`, the download half; the `from_pretrained` *constructor*
  that used to sit on top of it went with the config layer (`REQUIRED_FOR_V1.md` §5).

- **bpe** / **unigram** / **wordpiece** / **wordlevel**: one per model. Only `bpe` is on by
  default, because none of the current SOTA models use anything else. A `tokenizer.json` naming
  a model whose feature is off is refused at load rather than mis-read.

- **normalizers**: the table-backed normalizers (NFC/NFD/NFKC/NFKD, `Nmt`, `StripAccents`,
  `Bert`, SentencePiece's precompiled charsmap) and the ~150 KB of static Unicode tables behind
  them.

- **unicode-scripts**: the `UnicodeScripts` pre-tokenizer. Its script table is a 64 KiB
  `LazyLock` plus ~19 KB of generated range matching, and no widely-used `tokenizer.json`
  declares it — hence off by default.

- **parallelism**: rayon-backed batch encoding.

- **fancy-regex**: the optional system-regex backend, needed *only* for a genuine regex pattern
  in a `Split` pre-tokenizer or a `Replace` normalizer. The bitsplit-native pre-tokenizers
  (GPT-2, cl100k, o200k, tekken, deepseek, the class family, char-delimiter) need no backend,
  and a literal pattern is searched for directly.

## Building small

The default build is the slim one: no serde at all, and only BPE.
`make slim-size` prints its stripped and gzipped size.

The largest remaining cost is not in this crate: `std`'s panic, unwinding and
backtrace-symbolisation machinery (`addr2line`, `gimli`, `rustc_demangle`, `object`, plus the
build machine's source paths baked into `__cstring`) is about 40% of the gzipped binary. A stable
toolchain cannot drop it; a nightly one can:

```
cargo +nightly build --profile minsize \
  -Z build-std=std,panic_abort \
  -Z build-std-features=panic_immediate_abort ...
```

Measured, same features and profile: 360,045 -> 213,744 bytes gzipped (-40.6%). The trade is
real: every panic becomes an immediate `abort` with no message, unwinding or backtrace, so a
panicking input looks like any other crash. Usually fine on-device, rarely fine on a server.
Worth pairing with `--remap-path-prefix` — not for size (~175 bytes compressed) but because a
shipped binary should not contain a developer's home directory.

License: Apache-2.0

### Repeated borrowed batches

`PipelineTokenizer::encode_batch_into` accepts `&[&str]` and reuses a caller-owned
`Vec<pipeline::Encoding>`. It returns rows in input order and applies configured
padding. `encode_batch_for_each` instead visits each full encoding on its worker,
allowing the consumer to process results without materializing the entire batch.
Callbacks receive the input index, may run concurrently and out of order, and must
copy anything they retain. Configured padding requires temporary batch storage
and is applied before callbacks. Both methods accept single-sequence inputs;
use `encode` for sequence pairs or an asynchronous handle.

With the `parallelism` feature, repeated short batches can benefit from keeping
workers active briefly between calls:

```rust
# #[cfg(feature = "parallelism")]
# {
use std::time::Duration;
use tk_encode::utils::parallelism;

parallelism::set_num_threads(8);
parallelism::set_idle_spin_timeout(Duration::from_micros(200));
# }
```

Idle spinning is opt-in and defaults to zero. While active, workers execute queued
Rayon work cooperatively; after the timeout without work, they return to Rayon's
normal scheduling. This trades idle CPU consumption after a burst for fewer OS
yields and wakeups. The appropriate timeout depends on workload and hardware;
set `Duration::ZERO` to disable it. It does not change the software word cache or
promise a particular scaling factor.

The `tk-serialize` example `hot_batch` measures repeated, warmed batches while
copying every result's token IDs on the worker. It verifies complete encodings
against `encode(...).wait()` outside timing and emits an output hash:

```sh
cargo run --release -p tk-serialize --example hot_batch \
  --features 'deserialize parallelism bpe fancy-regex normalizers' -- \
  /path/to/tokenizer.json /path/to/corpus.txt 8 200
```

Run with `1 0` and `8 0` as controls. This example deliberately measures hot
repetition of the same 100 UTF-8 chunks, rather than first-time encoding.
