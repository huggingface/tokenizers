//! Size probe: the `.tok` read path, structurally identical to `binsize_pipeline.rs` so the
//! stripped sizes compare like for like. This is what a serving binary actually links.

use tk_encode::pipeline::PipelineTokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: binsize_tok <tokenizer.tok> <text>");
    let text = args.next().expect("usage: binsize_tok <tokenizer.tok> <text>");
    let file = tk_serialization::TokFile::open(path).unwrap();
    let tok = PipelineTokenizer::from_tok(file.bytes()).unwrap();
    println!("{}", tok.encode(text.as_str(), false).unwrap().len());
}
