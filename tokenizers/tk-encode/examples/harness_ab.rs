//! Why does tokbench report ~200 MB/s where the in-repo benches report ~1100?
//!
//! Same binary, same build, same corpus, same tokenizer. Only the harness shape changes:
//!
//!   A  one long input, output buffer reserved once and reused   (ab_giga / fixture_bench shape)
//!   B  one long input, fresh output per call                    (A + the per-call allocation)
//!   C  10 kB chunks, fresh output per call                      (B + chunking)
//!   D  10 kB chunks, fresh output per call, ids copied out      (C + tokbench's adapter copy)
//!
//! D is what tokbench measures. The ratio A/D is the harness effect with the code held fixed.
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const CHUNK: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;
const PASSES: usize = 4;

fn best(mut f: impl FnMut() -> (usize, usize)) -> (f64, usize) {
    let mut best = 0.0f64;
    let mut tokens = 0;
    for _ in 0..PASSES {
        let start = Instant::now();
        let (bytes, tok) = f();
        let secs = start.elapsed().as_secs_f64();
        tokens = tok;
        best = best.max(bytes as f64 / secs / 1e6);
    }
    (best, tokens)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: harness_ab <model.json> <corpus.txt>...");
    let legacy = Tokenizer::from_file(&model).expect("load model");
    let pipe = PipelineTokenizer::try_from(&legacy).expect("build pipeline");

    println!(
        "{:<10} {:>9} {:>9} {:>9} {:>9}   {:>7}",
        "corpus", "A long", "B +alloc", "C +chunk", "D +idcopy", "A/D"
    );
    for path in args {
        let text = std::fs::read_to_string(&path).expect("read corpus");
        // tokbench sees only the first MAX_CHUNKS*CHUNK bytes, so give every mode the same bytes.
        let mut end = (CHUNK * MAX_CHUNKS).min(text.len());
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        let text = &text[..end];
        let chunks: Vec<&str> = {
            let mut v = Vec::new();
            let mut s = 0;
            while s < text.len() {
                let mut e = (s + CHUNK).min(text.len());
                while e < text.len() && !text.is_char_boundary(e) {
                    e += 1;
                }
                v.push(&text[s..e]);
                s = e;
            }
            v
        };
        let bytes = text.len();

        // A: one long input, buffer reserved once and reused across passes.
        let mut reused = Vec::with_capacity(bytes / 3 + 16);
        let (a, tok_a) = best(|| {
            reused.clear();
            pipe.encode_generic_into::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(
                text, false, &mut reused,
            )
            .unwrap();
            black_box(&reused);
            (bytes, reused.len())
        });

        // B: one long input, the public allocating entry point.
        let (b, tok_b) = best(|| {
            let out = pipe
                .encode_generic::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(text, false)
                .unwrap();
            black_box(&out);
            (bytes, out.len())
        });

        // C: 10 kB chunks, allocating entry point.
        let (c, tok_c) = best(|| {
            let mut n = 0;
            for ch in &chunks {
                let out = pipe
                    .encode_generic::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(ch, false)
                    .unwrap();
                black_box(&out);
                n += out.len();
            }
            (bytes, n)
        });

        // D: exactly tokbench: chunks, allocating call, then ids restated into the harness buffer.
        let mut ids: Vec<u32> = Vec::new();
        let (d, tok_d) = best(|| {
            ids.clear();
            for ch in &chunks {
                let out = pipe
                    .encode_generic::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(ch, false)
                    .unwrap();
                ids.extend(out.iter().map(|t| t.id));
            }
            black_box(&ids);
            (bytes, ids.len())
        });

        let name = std::path::Path::new(&path)
            .file_stem()
            .unwrap()
            .to_string_lossy();
        assert_eq!(tok_a, tok_b, "{name}: A and B must agree");
        assert_eq!(tok_c, tok_d, "{name}: C and D must agree");
        println!(
            "{name:<10} {a:>9.0} {b:>9.0} {c:>9.0} {d:>9.0}   {:>7.2}   [{tok_a} tok whole, {tok_c} tok chunked]",
            a / d
        );
    }
}
