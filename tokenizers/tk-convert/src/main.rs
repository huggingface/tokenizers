//! `tk-convert tokenizer.json [...]` — writes `<name>.tok` beside each input.
//!
//! Conversion runs once, offline, on a machine that already has the JSON stack. Nothing here is
//! reachable from a serving binary: that side calls `PipelineTokenizer::from_tok` and links no
//! parser at all.

use tk_encode::Tokenizer;
use tk_encode::tokenizer::tok::to_tok;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: tk-convert <tokenizer.json> [...]  # writes <tokenizer>.tok beside each");
        std::process::exit(2);
    }

    let mut failed = 0;
    for input in &args {
        let output = format!("{}.tok", input.trim_end_matches(".json"));
        match convert(input, &output) {
            Ok((before, after)) => println!(
                "{output}  {:.1} MB from {:.1} MB  ({:.2}x)",
                after as f64 / 1e6,
                before as f64 / 1e6,
                before as f64 / after.max(1) as f64,
            ),
            Err(e) => {
                eprintln!("{input}: {e}");
                failed += 1;
            }
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}

fn convert(input: &str, output: &str) -> Result<(u64, usize), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_file(input)?;
    let bytes = to_tok(&tokenizer)?;
    std::fs::write(output, &bytes)?;
    Ok((std::fs::metadata(input)?.len(), bytes.len()))
}
