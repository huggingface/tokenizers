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
`bitsplit` SIMD pre-tokenizer, and the shared `bitmap_gen` tables):

- [`tk_encode`] — inference: the model engines and the full pipeline components
  ([`Normalizer`], [`PreTokenizer`], [`Model`], [`PostProcessor`], [`Decoder`]).
- `tk_serialize` — the reader: `from_json_file` turns a canonical `tokenizer.json` into a
  [`pipeline::PipelineTokenizer`], with no serde anywhere.
- [`tk_convert`] — the upgrade pass: [`canonicalize_file`] rewrites a `tokenizer.json` written by
  an older version into the canonical form that reader accepts.

This `tokenizers` crate is a thin umbrella that re-exports them so existing `tokenizers::…`
paths keep working.

### What rc0 does not have

The `Tokenizer` object model — `Tokenizer::new`, the component setters, `add_tokens`,
`save`, `from_pretrained`, truncation and padding — and the trainers are **not** in this
release. [`pipeline::PipelineTokenizer`] is read-only: it encodes and decodes what a
`tokenizer.json` describes and has no way to be built up or written back out. See
`REQUIRED_FOR_V1.md` at the repository root for the full list and why each one is deferred.

### Tokenization example

Read a config and encode with it. The example lives in `tk-serialize`, which is the only crate
that can compile it.
