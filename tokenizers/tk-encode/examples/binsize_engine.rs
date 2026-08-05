//! Size probe: the encode engine with **no config parser reachable**.
//!
//! Structurally identical to `binsize_pipeline.rs` except that the model is built in code instead
//! of read from a `tokenizer.json`, so `Tokenizer::from_file` — and with it `serde_json` and
//! everything only the JSON path reaches — is dead and LTO may drop it. The gap between the two
//! stripped binaries is what a load-free format can actually save.

use tk_encode::models::bpe::{BPE, PipelineBPE};
use tk_encode::pipeline::Model;

fn main() {
    let mut args = std::env::args().skip(1);
    let text = args.next().expect("usage: binsize_engine <text>");

    // A vocabulary big enough that nothing folds away, built without a parser.
    let mut vocab = tk_encode::models::bpe::Vocab::default();
    for b in 0u8..=255 {
        vocab.insert(format!("<{b:#04X}>"), b as u32);
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .unwrap();
    let model = PipelineBPE::from_bpe(bpe, false).unwrap();

    let mut scratch = model.init_scratch();
    let mut out = Vec::new();
    model.tokenize_pipeline(&text, &mut scratch, &mut out).unwrap();
    println!("{}", out.len());
}
