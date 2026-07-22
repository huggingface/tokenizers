# tk-encode

The core of `tokenizers`, written in Rust.
Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

## What is a Tokenizer

A Tokenizer works as a pipeline, it processes some raw text as input and outputs an `Encoding`.
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

### Loading a pretrained tokenizer from the Hub
```rust
use tk_encode::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    // needs http feature enabled
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;

    let encoding = tokenizer.encode("Hey there!", false)?;
    println!("{:?}", encoding.get_tokens());
    Ok(())
}
```

### Deserialization and tokenization example

```rust
use tk_encode::tokenizer::{Result, Tokenizer, EncodeInput};
use tk_encode::models::bpe::BPE;

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

Training lives in the companion `tk-train` crate (re-exported by the
`tokenizers` umbrella crate behind the `train` feature).

## Additional information

- tokenizers is designed to leverage CPU parallelism when possible. The level of parallelism is determined
  by the total number of core/threads your CPU provides but this can be tuned by setting the `RAYON_RS_NUM_THREADS`
  environment variable. As an example setting `RAYON_RS_NUM_THREADS=4` will allocate a maximum of 4 threads.
  **_Please note this behavior may evolve in the future_**

## PipelineTokenizer: oracle tests & benchmark

`PipelineTokenizer` is an experimental, allocation-light re-implementation of the
encode/decode pipeline. Its correctness is judged against the **latest released
`tokenizers` crate** (not the in-tree `Tokenizer`, which is being retired): the
pipeline must produce identical token ids and identical decoded text.

That comparison lives behind the optional `bench-baseline` feature (it links the
released crate), so a plain `cargo test` skips it. To run it you need the fixture
corpora and model tokenizers, then the feature flag:

```bash
# from tokenizers/ — fetches data/fixtures/ + model tokenizers (needs HF_TOKEN)
make fixtures bench-models

# encode parity (pipeline ids == released `encode_fast` ids)
cargo test -p tk-encode --features bench-baseline --test pipeline_oracle

# decode parity (pipeline decode == released `decode`) — ignored until
# PipelineTokenizer::decode is implemented, so pass --ignored to run it
cargo test -p tk-encode --features bench-baseline --test pipeline_decode_oracle -- --ignored
```

Both oracles sample seeded-random windows of every fixture corpus; a model whose
tokenizer file isn't present is skipped. CI runs them in the **Pipeline
Benchmark** workflow (`.github/workflows/pipeline-bench.yml`).

The same workflow runs the comparative benchmark — throughput, thread scaling,
memory, and binary size vs the release — and renders it to charts. To reproduce
locally:

```bash
cargo run --release -p tk-encode --features bench-baseline --example fixture_bench > bench.json
python3 ../.github/scripts/render_pipeline_bench.py bench.json   # writes SVGs + pipeline_bench.md
```

## Features

- **progressbar**: The progress bar visualization is enabled by default. It might be disabled if
  compilation for certain targets is not supported by the [termios](https://crates.io/crates/termios)
  dependency of the [indicatif](https://crates.io/crates/indicatif) progress bar.

- **http**: This feature enables downloading the tokenizer via HTTP. It is disabled by default.
  With this feature enabled, `Tokenizer::from_pretrained` becomes accessible.

License: Apache-2.0
