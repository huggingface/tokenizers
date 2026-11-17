<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg">
    <a href="https://github.com/huggingface/tokenizers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue">
    </a>
    <a href="https://docs.rs/tokenizers/">
        <img alt="Doc" src="https://docs.rs/tokenizers/badge.svg">
    </a>
</p>
<br>


The 🤗 Tokenizers library.

The implementation is split across crates (each built on internal engines — `tk_encode` on the
`bitcannon` SIMD pre-tokenizer, and the shared `bitmap_gen` tables):

- [`tk_encode`] — inference: the model engines and the full pipeline components
  ([`Normalizer`], [`PreTokenizer`], [`Model`], [`PostProcessor`], [`Decoder`]).
- `tk_serialize` — the reader: `from_json_file` turns a canonical `tokenizer.json` into a
  [`pipeline::Tokenizer`], with no serde anywhere.
- [`tk_convert`] — the upgrade pass: [`canonicalize_file`] rewrites a `tokenizer.json` written by
  an older version into the canonical form that reader accepts.

This `tokenizers` crate is a thin umbrella that re-exports them so existing `tokenizers::…`
paths keep working.

### Using the rust crate

Add it to your `Cargo.toml`:

```toml
[dependencies]
tokenizers = "1.0.0"
```

The default features cover inference: every model, normalizer and pre-tokenizer, batch encoding
across threads, and writing a tokenizer back out as JSON.

Downloading from the Hugging Face Hub is opt-in:

```toml
tokenizers = { version = "1.0.0", features = ["http"] }
```

Training is not in this crate yet, because its API will still change. To train a model,
depend on the `tk-train` crate (versioned 0.x) directly.

For a smaller build, turn the defaults off and keep only what your `tokenizer.json` needs.
A config that names a component you compiled out fails at load.

```toml
tokenizers = { version = "1.0.0", default-features = false, features = ["bpe"] }
```

| Feature | Default | What it turns on |
|---|---|---|
| `bpe` | yes | The BPE model. |
| `unigram` | yes | The Unigram model. |
| `wordpiece` | yes | The WordPiece model. |
| `wordlevel` | yes | The WordLevel model. |
| `normalizers` | yes | The normalizers that need Unicode tables: `NFC`, `NFD`, `NFKC`, `NFKD`, `Nmt`, `BertNormalizer`, `Precompiled` and `StripAccents`. The others are always built. |
| `unicode-scripts` | yes | The `UnicodeScripts` pre-tokenizer. |
| `parallelism` | yes | Batch encoding across threads, with rayon. |
| `serialize` | yes | Writing a tokenizer back out as `tokenizer.json` (`to_json`). Reading is always built. |
| `http` | no | `utils::from_pretrained`, which downloads a `tokenizer.json` from the Hugging Face Hub. |
| `regex` | no | A regex engine (`fancy-regex`) for `Split` and `Replace` regex patterns that the built-in pre-tokenizer does not cover. Without it, those configs fail at load. Plain-string patterns work either way. |

### What rc0 does not have

The `Tokenizer` object model — `Tokenizer::new`, the component setters, `add_tokens`,
`save`, `from_pretrained`, truncation and padding — is **not** in this release.
[`pipeline::Tokenizer`] is read-only: it encodes and decodes what a `tokenizer.json`
describes and has no way to be built up or written back out. See
`REQUIRED_FOR_V1.md` at the repository root for the full list and why each one is deferred.

### Tokenization example

Read a config and encode with it. The example lives in `tk-serialize`, which is the only crate
that can compile it.
