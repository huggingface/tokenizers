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


The core of `tokenizers`, written in Rust.
Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

## What is a Tokenizer

A Tokenizer works as a pipeline, it processes some raw text as input and outputs an `Encoding`.
The various steps of the pipeline are:

1. The `Normalizer`: in charge of normalizing the text. Common examples of normalization are
   the [unicode normalization standards](https://unicode.org/reports/tr15/#Norm_Forms), such as `NFD` or `NFKC`.
2. The `PreTokenizer`: in charge of creating initial words splits in the text. The most common way of
   splitting text is simply on whitespace.
3. The `Model`: in charge of doing the actual tokenization. An example of a `Model` would be
   `BPE` or `WordPiece`.
4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything relevant
   that, for example, a language model would need, such as special tokens.

## Quick example

```Rust
use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput};
use tokenizers::models::bpe::BPE;

fn main() -> Result<()> {
    let bpe_builder = BPE::from_files("./path/to/vocab.json", "./path/to/merges.txt")?;
    let bpe = bpe_builder
        .dropout(0.1)
        .unk_token("[UNK]".into())
        .build()?;

    let mut tokenizer = Tokenizer::new(Box::new(bpe));

    let encoding = tokenizer.encode(EncodeInput::Single("Hey there!".into()))?;
    println!("{:?}", encoding.get_tokens());

    Ok(())
}
```

## Additional information

- tokenizers is designed to leverage CPU parallelism when possible. The level of parallelism is determined
by the total number of core/threads your CPU provides but this can be tuned by setting the `RAYON_RS_NUM_CPUS`
environment variable. As an example setting `RAYON_RS_NUM_CPUS=4` will allocate a maximum of 4 threads.
**_Please note this behavior may evolve in the future_**
