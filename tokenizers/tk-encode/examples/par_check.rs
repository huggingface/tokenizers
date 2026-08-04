//! Does the parallel `encode` actually beat the serial `encode_one` on one big document,
//! and on a batch? Prints MB/s for each so a merge can't silently drop the parallelism.
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const DOC: usize = 8 * 1024 * 1024;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data");
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("MB/s, {} MB document, {threads} threads available\n", DOC >> 20);
    println!(
        "  {:<24} {:<9} {:>10} {:>10} {:>10} {:>7}",
        "model", "corpus", "serial", "par 1 doc", "par batch", "speedup"
    );
    for model in ["gpt2.json", "llama-3-tokenizer.json", "bert-wiki.json"] {
        let path = root.join(model);
        if !path.exists() {
            continue;
        }
        let legacy = Tokenizer::from_file(&path).expect("load");
        let Ok(pipe) = PipelineTokenizer::try_from(&legacy) else {
            continue;
        };
        for corpus in ["english", "chinese"] {
            let Ok(text) =
                std::fs::read_to_string(root.join("corpora").join(format!("{corpus}.txt")))
            else {
                continue;
            };
            let one = text.repeat(DOC.div_ceil(text.len()));
            let n = one.len();
            let mbs = |elapsed: f64| n as f64 / (1024.0 * 1024.0) / elapsed;

            let time = |f: &mut dyn FnMut()| {
                f();
                let t = Instant::now();
                for _ in 0..3 {
                    f();
                }
                t.elapsed().as_secs_f64() / 3.0
            };

            let serial = time(&mut || {
                black_box(pipe.encode_one(&one, true).unwrap());
            });
            let par_one = time(&mut || {
                black_box(pipe.encode(one.as_str()).wait_for_completion().unwrap());
            });
            // Same total bytes, cut into 16 inputs: the batch-parallel face.
            let chunk = n / 16;
            let mut parts: Vec<&str> = Vec::new();
            let mut at = 0;
            while at < n {
                let mut end = (at + chunk).min(n);
                while end < n && !one.is_char_boundary(end) {
                    end += 1;
                }
                parts.push(&one[at..end]);
                at = end;
            }
            let par_batch = time(&mut || {
                black_box(pipe.encode(&parts[..]).wait_for_completion().unwrap());
            });

            println!(
                "  {model:<24} {corpus:<9} {:>10.0} {:>10.0} {:>10.0} {:>6.1}x",
                mbs(serial),
                mbs(par_one),
                mbs(par_batch),
                serial / par_one.min(par_batch),
            );
        }
    }
}
