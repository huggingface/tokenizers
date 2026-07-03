//! Comparative throughput of the reference `Tokenizer` vs the experimental
//! `PipelineTokenizer` over every corpus in `data/fixtures/` (languages +
//! modalities), on ~10 kB inputs — the regime where per-input overhead is
//! amortized (see `pipeline_benchmark.rs` for the size sweep).
//!
//! Emits one JSON object per fixture on stdout, consumed by
//! `.github/scripts/render_pipeline_bench.py` in CI.

use std::convert::TryFrom;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;
const REPS: usize = 5;

fn make_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
        if cur.len() >= CHUNK_BYTES {
            chunks.push(std::mem::take(&mut cur));
            if chunks.len() == MAX_CHUNKS {
                return chunks;
            }
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

fn median_secs(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn time_pass(encode: &dyn Fn(&str) -> usize, chunks: &[String]) -> f64 {
    let start = Instant::now();
    let mut n = 0usize;
    for chunk in chunks {
        n += encode(chunk);
    }
    black_box(n);
    start.elapsed().as_secs_f64()
}

fn fixture_files() -> Vec<(String, PathBuf)> {
    let mut files = Vec::new();
    for group in ["lang", "modalities"] {
        let dir = Path::new(DATA_DIR).join("fixtures").join(group);
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e} — run `make fixtures` first", dir.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "txt"))
            .collect();
        entries.sort();
        for path in entries {
            files.push((group.to_string(), path));
        }
    }
    files
}

fn main() {
    let oracle = Tokenizer::from_file(format!("{DATA_DIR}/bert-wiki.json")).unwrap();
    let pipeline = PipelineTokenizer::try_from(&oracle).unwrap();

    let legacy_enc = |s: &str| oracle.encode(s, false).unwrap().len();
    let pipeline_enc = |s: &str| pipeline.encode(s, false).unwrap().len();

    println!("[");
    let files = fixture_files();
    for (i, (group, path)) in files.iter().enumerate() {
        let name = path.file_stem().unwrap().to_str().unwrap().to_string();
        let text = std::fs::read_to_string(path).unwrap();
        let chunks = make_chunks(&text);
        let bytes: usize = chunks.iter().map(String::len).sum();

        let ids_match = chunks.iter().take(3).all(|c| {
            let expected = oracle.encode(c.as_str(), false).unwrap();
            let got: Vec<u32> = pipeline
                .encode(c, false)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect();
            expected.get_ids() == got
        });

        // interleave impls so frequency/thermal drift hits both equally
        time_pass(&legacy_enc, &chunks);
        time_pass(&pipeline_enc, &chunks);
        let (mut legacy_s, mut pipeline_s) = (Vec::new(), Vec::new());
        for _ in 0..REPS {
            legacy_s.push(time_pass(&legacy_enc, &chunks));
            pipeline_s.push(time_pass(&pipeline_enc, &chunks));
        }
        let (l, p) = (median_secs(legacy_s), median_secs(pipeline_s));

        eprintln!(
            "{name}: legacy {:.1} MB/s, pipeline {:.1} MB/s",
            bytes as f64 / l / 1e6,
            bytes as f64 / p / 1e6
        );
        println!(
            "{}{}",
            serde_json::json!({
                "fixture": name,
                "group": group,
                "bytes": bytes,
                "chunks": chunks.len(),
                "legacy_mbps": bytes as f64 / l / 1e6,
                "pipeline_mbps": bytes as f64 / p / 1e6,
                "speedup": l / p,
                "ids_match": ids_match,
            }),
            if i + 1 < files.len() { "," } else { "" }
        );
    }
    println!("]");
}
