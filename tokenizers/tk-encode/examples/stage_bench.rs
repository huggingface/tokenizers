//! Where the encode time goes, via the pipeline's own ablation ladder: each level runs every
//! stage up to itself, so a stage's marginal cost is the difference between two levels.
//!
//! Run with the cache on and off to see what the WordCache actually removes.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use tk_encode::pipeline::{Model, PipelineTokenizer, Span};
use tk_encode::{ModelWrapper, Tokenizer};

const MB: usize = 4 * 1024 * 1024;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data");
    let models = ["gpt2.json", "llama-3-tokenizer.json"];
    let corpora = ["english", "code", "chinese", "russian"];

    println!("marginal ns/B per stage (4 MB input, warm, single thread)\n");
    for model in models {
        let path = root.join(model);
        if !path.exists() {
            continue;
        }
        println!("{model}");
        println!(
            "  {:<9} {:>6} {:>8} {:>7} {:>7} {:>7} {:>7}  {:>9}",
            "corpus", "cache", "frame", "normal", "split", "model", "post", "total MB/s"
        );
        for corpus in corpora {
            let Ok(text) = std::fs::read_to_string(root.join("corpora").join(format!("{corpus}.txt")))
            else {
                continue;
            };
            let one = text.repeat(MB.div_ceil(text.len()));
            let n = one.len();
            let reps = (400_000_000 / n as u64).clamp(2, 20) as u32;

            let cap: usize = std::env::var("TK_CACHE").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000);
            for capacity in [cap] {
                let mut tok = Tokenizer::from_file(&path).expect("load");
                if let ModelWrapper::BPE(bpe) = tok.get_model_mut() {
                    bpe.resize_cache(capacity);
                }
                let pipe = PipelineTokenizer::try_from(&tok).expect("pipeline");
                let mut pre = Vec::<Span>::new();
                let mut scratch = pipe.get_model().init_scratch();
                let mut out = Vec::new();

                macro_rules! level {
                    ($s:expr) => {{
                        for _ in 0..2 {
                            pre.clear();
                            out.clear();
                            pipe.encode_generic::<$s>(&one, false, &mut pre, &mut scratch, &mut out)
                                .unwrap();
                        }
                        let t = Instant::now();
                        for _ in 0..reps {
                            pre.clear();
                            out.clear();
                            pipe.encode_generic::<$s>(&one, false, &mut pre, &mut scratch, &mut out)
                                .unwrap();
                            black_box((&pre, &out));
                        }
                        t.elapsed().as_secs_f64() / f64::from(reps) * 1e9 / n as f64
                    }};
                }
                let frame = level!(0);
                let norm = level!(1);
                let split = level!(2);
                let model_l = level!(3);
                let post = level!(4);
                println!(
                    "  {corpus:<9} {:>6} {:>8.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2}  {:>9.0}",
                    if capacity == 0 { "off" } else { "on" },
                    frame,
                    norm - frame,
                    split - norm,
                    model_l - split,
                    post - model_l,
                    1e3 / post,
                );
            }
        }
        println!();
    }
}
