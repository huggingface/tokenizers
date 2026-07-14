//! Minimal encode program measured by CI for binary size: the `PipelineTokenizer`
//! encode path (which still links the `Tokenizer` build path — a pipeline is
//! built from one). Kept structurally identical to `binsize_baseline.rs` so the
//! stripped sizes compare like for like.

use std::convert::TryFrom;

use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: binsize_pipeline <tokenizer.json> <text>");
    let text = args
        .next()
        .expect("usage: binsize_pipeline <tokenizer.json> <text>");
    let tok = Tokenizer::from_file(path).unwrap();
    let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
    println!(
        "{}",
        pipeline.encode_scoped(text.as_str(), false, |h| h.into_single()).unwrap().len()
    );
}
