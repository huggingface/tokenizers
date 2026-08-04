//! Multi-thread encode: the legacy in-tree `Tokenizer` against the pipeline, one 4 MB document
//! per thread, at a sweep of thread counts. Aggregate MB/s across all threads.
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;
use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const DOC: usize = 4 * 1024 * 1024;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data");
    let maxt = std::thread::available_parallelism().map_or(1, |n| n.get());
    let counts: Vec<usize> = [1usize, 2, 4, 8, maxt]
        .into_iter()
        .filter(|&t| t <= maxt)
        .collect();
    println!(
        "aggregate MB/s, one {} MB doc per thread\n",
        DOC / (1024 * 1024)
    );
    for model in ["gpt2.json", "llama-3-tokenizer.json"] {
        let path = root.join(model);
        if !path.exists() {
            continue;
        }
        for corpus in ["english", "chinese"] {
            let Ok(text) =
                std::fs::read_to_string(root.join("corpora").join(format!("{corpus}.txt")))
            else {
                continue;
            };
            let one = text.repeat(DOC.div_ceil(text.len()));
            let bytes = one.len();
            let legacy = Tokenizer::from_file(&path).expect("load");
            let pipe = PipelineTokenizer::try_from(&legacy).expect("pipeline");
            print!("{model:<24} {corpus:<8}");
            for &t in &counts {
                let run = |f: &(dyn Fn() + Sync)| {
                    // warm, then time threads spawned once
                    std::thread::scope(|s| {
                        for _ in 0..t {
                            s.spawn(f);
                        }
                    });
                    let start = Instant::now();
                    std::thread::scope(|s| {
                        for _ in 0..t {
                            s.spawn(|| {
                                for _ in 0..3 {
                                    f();
                                }
                            });
                        }
                    });
                    (bytes * t * 3) as f64 / start.elapsed().as_secs_f64() / 1e6
                };
                let l = run(&|| {
                    black_box(legacy.encode_fast(one.as_str(), false).unwrap());
                });
                let p = run(&|| {
                    black_box(pipe.encode(one.as_str(), false).unwrap());
                });
                print!("  │ {t:>2}t legacy {l:>5.0} ours {p:>6.0}");
            }
            println!();
        }
    }
}
