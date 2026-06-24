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

Starting with `0.23`, the implementation is split across two crates:

- [`tk_encode`] — inference: the model engines, the full pipeline components
  ([`Normalizer`], [`PreTokenizer`], [`Model`], [`PostProcessor`],
  [`Decoder`]) and the [`Tokenizer`] orchestration (encode / decode).
- [`tk_train`] — training: the [`Trainer`] trait, every concrete `*Trainer`,
  and the [`TokenizerTrainExt`] extension that adds `train` /
  `train_from_files` onto a [`Tokenizer`].

This `tokenizers` crate is a thin umbrella that re-exports both so existing
`tokenizers::…` paths keep working. Training lives behind the (default-on)
`train` feature; disable default features for an inference-only build.

### Deserialization and tokenization example

```rust
use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput};
use tokenizers::models::bpe::BPE;

fn main() -> Result<()> {
    let bpe_builder = BPE::from_file("./path/to/vocab.json", "./path/to/merges.txt");
    let bpe = bpe_builder
        .dropout(0.1)
        .unk_token("[UNK]".into())
        .build()?;

    let mut tokenizer = Tokenizer::new(bpe);

    let encoding = tokenizer.encode("Hey there!", false)?;
    println!("{:?}", encoding.get_tokens());

    Ok(())
}
```
