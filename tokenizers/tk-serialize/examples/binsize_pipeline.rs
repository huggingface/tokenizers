//! Minimal encode program measured by CI for binary size: the `PipelineTokenizer` encode path.
//! Kept structurally identical to `binsize_baseline.rs` so the stripped sizes compare like for
//! like. Built WITHOUT `bench-baseline` (unlike the baseline example), so the size is tk-encode's
//! real shipping footprint — none of the benchmark-only reference-regex deps.
//!
//! One read path only, the hand-rolled reader: the config layer lives in another crate, so there is
//! no `Tokenizer` here to deserialize into.

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: binsize_pipeline <tokenizer.json> <text>");
    let text = args
        .next()
        .expect("usage: binsize_pipeline <tokenizer.json> <text>");
    let pipeline = tk_serialize::from_json_file(path).unwrap();
    println!(
        "{}",
        pipeline
            .encode(text.as_str(), false)
            .wait()
            .unwrap()
            .first()
            .unwrap()
            .len()
    );
}
