//! Compare the reference `Tokenizer` (the oracle) against the experimental
//! `PipelineTokenizer` for correctness (token ids + byte offsets) and
//! throughput, across two input regimes:
//!
//! * LINES  — one short sequence per line (per-input overhead dominates)
//! * CHUNKS — lines packed into ~8 KB documents (long sequences; per-input
//!   overhead is amortized over many tokens)
//!
//! Usage:
//!   cargo run --release --example pipeline_compare -- [tokenizer.json] [text_file] [repeats]
//!
//! Defaults (run from the `tk-encode` crate dir): the trained bert-wiki
//! tokenizer (`Whitespace` pre-tokenizer + `WordPiece`) over `big.txt`, 3 timed
//! passes. The oracle is always called with `add_special_tokens = false` so the
//! post-processor (which the pipeline does not yet apply) cannot cause spurious
//! mismatches.

use std::convert::TryFrom;
use std::time::Instant;

use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Target size of a "long sequence" document in the CHUNKS regime.
const CHUNK_TARGET_BYTES: usize = 8 * 1024;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let tokenizer_path = args
        .next()
        .unwrap_or_else(|| "../data/bert-wiki.json".into());
    let text_path = args.next().unwrap_or_else(|| "../data/big.txt".into());
    let repeats: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(3);

    println!("tokenizer : {tokenizer_path}");
    println!("corpus    : {text_path}");
    println!();

    let oracle = Tokenizer::from_file(&tokenizer_path)?;
    let pipeline = PipelineTokenizer::try_from(&oracle)?;

    let text = std::fs::read_to_string(&text_path)?;

    // Regime 1: one short sequence per non-empty line.
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    // Regime 2: pack consecutive lines into ~CHUNK_TARGET_BYTES documents.
    let chunks: Vec<String> = make_chunks(&lines, CHUNK_TARGET_BYTES);
    let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();

    run_regime(
        "LINES  (short sequences)",
        &oracle,
        &pipeline,
        &lines,
        "line",
        repeats,
    )?;
    println!();
    run_regime(
        "CHUNKS (long sequences)",
        &oracle,
        &pipeline,
        &chunk_refs,
        "chunk",
        repeats,
    )?;

    Ok(())
}

/// Pack consecutive lines (joined with `\n`) into documents of at least
/// `target_bytes`, so the same corpus can be encoded as long sequences.
fn make_chunks(lines: &[&str], target_bytes: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in lines {
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
        if cur.len() >= target_bytes {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Print a header then run correctness + performance for one input regime.
fn run_regime(
    name: &str,
    oracle: &Tokenizer,
    pipeline: &PipelineTokenizer,
    inputs: &[&str],
    unit: &str,
    repeats: usize,
) -> Result<()> {
    let total_bytes: usize = inputs.iter().map(|s| s.len()).sum();
    let avg = total_bytes / inputs.len().max(1);
    println!("######## {name} ########");
    println!(
        "  {} inputs, {:.2} MB, avg {avg} B/input",
        inputs.len(),
        total_bytes as f64 / 1e6
    );
    correctness(oracle, pipeline, inputs, unit)?;
    performance(oracle, pipeline, inputs, total_bytes, repeats)?;
    Ok(())
}

/// Encode every input with both tokenizers and compare ids, then (where ids
/// match) the per-token byte offsets.
fn correctness(
    oracle: &Tokenizer,
    pipeline: &PipelineTokenizer,
    inputs: &[&str],
    unit: &str,
) -> Result<()> {
    let mut id_mismatches = 0usize;
    let mut offset_mismatches = 0usize;
    let mut shown_ids = 0usize;
    let mut shown_offsets = 0usize;

    for input in inputs {
        let expected = oracle.encode(*input, false)?;
        let expected_ids = expected.get_ids();
        let expected_offsets = expected.get_offsets();

        let got = pipeline.encode(input, false)?;
        let got_ids: Vec<u32> = got.iter().map(|t| t.id).collect();

        if expected_ids != got_ids.as_slice() {
            id_mismatches += 1;
            if shown_ids < 5 {
                shown_ids += 1;
                let preview: String = input.chars().take(60).collect();
                let diff_at = expected_ids
                    .iter()
                    .zip(&got_ids)
                    .position(|(a, b)| a != b)
                    .unwrap_or(expected_ids.len().min(got_ids.len()));
                println!("  ID MISMATCH on {preview:?}");
                println!(
                    "    oracle   ({} ids): {:?}...",
                    expected_ids.len(),
                    &expected_ids[..expected_ids.len().min(diff_at + 6)]
                );
                println!(
                    "    pipeline ({} ids): {:?}...",
                    got_ids.len(),
                    &got_ids[..got_ids.len().min(diff_at + 6)]
                );
            }
            // ids differ => token counts differ => offsets aren't comparable
            continue;
        }

        // ids match: compare byte offsets token-for-token
        let got_offsets: Vec<(usize, usize)> = got
            .iter()
            .map(|t| (t.start as usize, t.end as usize))
            .collect();
        if expected_offsets != got_offsets.as_slice() {
            offset_mismatches += 1;
            if shown_offsets < 5 {
                shown_offsets += 1;
                let preview: String = input.chars().take(60).collect();
                let diff_at = expected_offsets
                    .iter()
                    .zip(&got_offsets)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                let end = expected_offsets.len().min(diff_at + 4);
                println!("  OFFSET MISMATCH on {preview:?} (first diff at token #{diff_at})");
                println!("    oracle  : {:?}", &expected_offsets[diff_at..end]);
                println!("    pipeline: {:?}", &got_offsets[diff_at..end]);
            }
        }
    }

    let n = inputs.len();
    let ids_ok = n - id_mismatches;
    let offsets_ok = ids_ok - offset_mismatches;
    println!("== correctness ==");
    println!(
        "  ids    : {ids_ok}/{n} {unit}s match ({id_mismatches} mismatch{})",
        plural(id_mismatches)
    );
    println!(
        "  offsets: {offsets_ok}/{ids_ok} id-matching {unit}s match ({offset_mismatches} mismatch{})",
        plural(offset_mismatches)
    );
    if id_mismatches == 0 && offset_mismatches == 0 {
        println!("  ✅ ids and offsets are identical to the oracle");
    } else if id_mismatches == 0 {
        println!(
            "  ✅ ids identical; ⚠️  offsets diverge (pipeline emits normalized-space offsets)"
        );
    }
    Ok(())
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "es"
    }
}

/// Time `repeats` sequential passes over the inputs with each tokenizer.
fn performance(
    oracle: &Tokenizer,
    pipeline: &PipelineTokenizer,
    inputs: &[&str],
    total_bytes: usize,
    repeats: usize,
) -> Result<()> {
    // Warm up (touch caches, fill any lazy statics) and prevent the optimizer
    // from eliding the encode calls by summing token counts.
    let mut sink = 0usize;
    for input in inputs {
        sink += oracle.encode(*input, false)?.len();
        sink += pipeline.encode(input, false)?.len();
    }

    let oracle_ns = time_pass(repeats, || {
        let mut n = 0usize;
        for input in inputs {
            n += oracle.encode(*input, false).unwrap().len();
        }
        n
    });

    let pipeline_ns = time_pass(repeats, || {
        let mut n = 0usize;
        for input in inputs {
            n += pipeline.encode(input, false).unwrap().len();
        }
        n
    });

    println!("== performance (single thread, best of {repeats}) ==");
    report("oracle  ", oracle_ns, total_bytes);
    report("pipeline", pipeline_ns, total_bytes);
    println!("  speedup : {:.2}x", oracle_ns as f64 / pipeline_ns as f64);
    std::hint::black_box(sink);
    Ok(())
}

/// Run `f` `repeats` times, returning the fastest wall-clock time in ns.
fn time_pass(repeats: usize, mut f: impl FnMut() -> usize) -> u128 {
    let mut best = u128::MAX;
    for _ in 0..repeats {
        let start = Instant::now();
        let n = f();
        let elapsed = start.elapsed().as_nanos();
        std::hint::black_box(n);
        best = best.min(elapsed);
    }
    best
}

fn report(label: &str, ns: u128, total_bytes: usize) {
    let secs = ns as f64 / 1e9;
    let mb_per_s = (total_bytes as f64 / 1e6) / secs;
    println!("  {label}: {secs:.4} s  ->  {mb_per_s:.1} MB/s");
}
