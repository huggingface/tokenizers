//! Convert each tokenizer to `.tok`, load it back through the read-only path, and prove the ids
//! are identical to what the JSON path produces.
//!
//! ```sh
//! cargo run --release -p tk-convert --example tok_check
//! ```

use std::convert::TryFrom;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;
use tk_convert::to_tok;

const CORPORA: &[&str] = &[
    "english", "chinese", "code", "dense", "russian", "arabic", "korean", "greek", "hindi", "thai",
];

const DEFAULT_MODELS: &[&str] = &[
    "data/gpt2.json",
    "data/roberta.json",
    "data/llama-3-tokenizer.json",
    "data/deepseek-v4.json",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<&str> = if args.is_empty() {
        DEFAULT_MODELS.to_vec()
    } else {
        args.iter().map(String::as_str).collect()
    };

    let texts: Vec<(&str, String)> = CORPORA
        .iter()
        .filter_map(|name| {
            std::fs::read_to_string(format!("data/corpora/{name}.txt"))
                .ok()
                .map(|t| (*name, t))
        })
        .collect();
    let units = models.len() * texts.len();
    println!(
        "{units} checks: {} models x {} corpora. Each = convert, reload from .tok, compare every id.\n",
        models.len(),
        texts.len()
    );

    let started = Instant::now();
    let (mut done, mut failures) = (0usize, 0usize);

    for path in &models {
        let json_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let reference = match Tokenizer::from_file(path).and_then(|t| {
            let packed = to_tok(&t)?;
            let pipeline = PipelineTokenizer::try_from(&t)?;
            Ok((pipeline, packed))
        }) {
            Ok(pair) => pair,
            Err(e) => {
                println!("{path}: {e}");
                failures += 1;
                continue;
            }
        };
        let (pipeline, packed) = reference;

        let tok_path = format!("{}.tok", path.trim_end_matches(".json"));
        std::fs::write(&tok_path, &packed).expect("write .tok");

        // Load through the same path an inference binary uses: aligned buffer, no parser.
        let t0 = Instant::now();
        let file = tk_serialization::TokFile::open(&tok_path).expect("open .tok");
        let loaded = match PipelineTokenizer::from_tok(file.bytes()) {
            Ok(p) => p,
            Err(e) => {
                println!("{path}: reload failed: {e}");
                failures += 1;
                continue;
            }
        };
        let load = t0.elapsed();

        println!(
            "{path}\n  json {:.1} MB  ->  .tok {:.1} MB ({:.2}x)   reload {:.1?}",
            json_bytes as f64 / 1e6,
            packed.len() as f64 / 1e6,
            json_bytes as f64 / packed.len() as f64,
            load,
        );

        for (name, text) in &texts {
            let want: Vec<u32> = pipeline
                .encode(text, true)
                .expect("reference encode")
                .iter()
                .map(|t| t.id)
                .collect();
            let got: Vec<u32> = loaded
                .encode(text, true)
                .expect(".tok encode")
                .iter()
                .map(|t| t.id)
                .collect();

            done += 1;
            let elapsed = started.elapsed();
            let eta = elapsed.mul_f64((units - done) as f64 / done as f64);

            if got == want {
                println!(
                    "  [{done}/{units}] {name:<8} ok  {:>7} ids   | elapsed {:.0?} eta {:.0?}",
                    want.len(),
                    elapsed,
                    eta
                );
            } else {
                failures += 1;
                let at = got
                    .iter()
                    .zip(&want)
                    .position(|(a, b)| a != b)
                    .unwrap_or(want.len().min(got.len()));
                println!(
                    "  [{done}/{units}] {name:<8} MISMATCH: {} ids vs {} expected, first differs at {at}: {:?} vs {:?}",
                    got.len(),
                    want.len(),
                    got.get(at),
                    want.get(at),
                );
            }
        }
        println!();
    }

    if failures == 0 {
        println!("all {units} checks byte-exact in {:.1?}", started.elapsed());
    } else {
        println!("{failures} FAILURES out of {units}");
        std::process::exit(1);
    }
}
