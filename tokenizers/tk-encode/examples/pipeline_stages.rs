//! Per-stage timing + allocation breakdown of `PipelineTokenizer::encode`.
//!
//! `encode` runs its stages inline, so to attribute time (and heap allocations)
//! to each step we reconstruct the exact same stages here from the tokenizer's
//! public components and wrap each region in a timer + an allocation counter (a
//! counting global allocator). The real `PipelineTokenizer::encode` is also run
//! as a ground-truth cross-check that the reconstruction captures ~all the work.
//!
//! Stages (mirroring the current `encode` + `tokenize_chunk`): special-token
//! extraction on the remaining input (the `get_next_special_token` scan loop)
//! -> normalization (+ NormalizedString build) -> pre-tokenization ->
//! tokenization (model.tokenize + extend output). Keep in sync with `encode`.
//!
//! Usage:
//!   cargo run --release --example pipeline_stages -- [tokenizer.json] [text_file] [repeats]

use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::TryFrom;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use tk_encode::pipeline::{
    PipelinePatternMatcher, PipelinePreTokenizer, PipelineToken, PipelineTokenizer,
    PreTokenizer as PipePreTok, Split,
};
use tk_encode::{
    AddedVocabulary, Model, ModelWrapper, NormalizedString, Normalizer, NormalizerWrapper,
    PreTokenizerWrapper, Token, Tokenizer,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const KB: usize = 1024;
/// Minimum bytes to process per regime/pass. If a corpus is smaller (or can't
/// form enough sequences of the target size), it is repeated to reach this.
const MIN_TOTAL_BYTES: usize = 2 * 1024 * 1024;

// ---- counting global allocator: tallies alloc/realloc/alloc_zeroed calls ----
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(l.size() as u64, Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(new.saturating_sub(l.size()) as u64, Relaxed);
        System.realloc(p, l, new)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(l.size() as u64, Relaxed);
        System.alloc_zeroed(l)
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

#[inline]
fn allocs() -> u64 {
    ALLOC_CALLS.load(Relaxed)
}

#[derive(Default, Clone, Copy)]
struct Acc {
    time: Duration,
    allocs: u64,
}
impl Acc {
    fn add(&mut self, t: Duration, a: u64) {
        self.time += t;
        self.allocs += a;
    }
}

#[derive(Default)]
struct Stages {
    extract: Acc,
    normalize: Acc,
    pretok: Acc,
    tokenize: Acc,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let tokenizer_path = args
        .next()
        .unwrap_or_else(|| "../data/bert-wiki.json".into());
    let text_path = args.next().unwrap_or_else(|| "../data/big.txt".into());
    let repeats: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(3);

    println!("tokenizer : {tokenizer_path}");
    println!("corpus    : {text_path}\n");

    let oracle = Tokenizer::from_file(&tokenizer_path)?;
    let pipeline = PipelineTokenizer::try_from(&oracle)?;
    let pretok = build_pipeline_pretokenizer(&oracle)?;

    let text = std::fs::read_to_string(&text_path)?;

    // Three sequence-length regimes. `None` = one short sequence per line.
    let regimes: [(&str, &str, Option<usize>); 3] = [
        ("SHORT (lines)", "short", None),
        ("LARGE (8 KB seqs)", "8kb", Some(8 * KB)),
        ("HUGE (128 KB seqs)", "128kb", Some(128 * KB)),
    ];

    for (name, tag, target) in regimes {
        // Repeat the corpus if it can't fill enough sequences of the target size.
        let min_total = target.map_or(MIN_TOTAL_BYTES, |t| (t * 8).max(MIN_TOTAL_BYTES));
        let mat = material(&text, min_total);
        let lines: Vec<&str> = mat.lines().filter(|l| !l.trim().is_empty()).collect();

        match target {
            None => run_regime(name, tag, &oracle, &pipeline, &pretok, &lines, repeats),
            Some(t) => {
                let chunks = make_chunks(&lines, t);
                let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
                run_regime(name, tag, &oracle, &pipeline, &pretok, &refs, repeats);
            }
        }
        println!();
    }

    Ok(())
}

/// Return `corpus` repeated (with newline separators) until it reaches at least
/// `min_bytes`; returns it as-is if already large enough.
fn material(corpus: &str, min_bytes: usize) -> String {
    if corpus.len() >= min_bytes {
        return corpus.to_string();
    }
    let mut s = String::with_capacity(min_bytes + corpus.len());
    while s.len() < min_bytes {
        s.push_str(corpus);
        if !corpus.ends_with('\n') {
            s.push('\n');
        }
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn run_regime(
    name: &str,
    tag: &str,
    oracle: &Tokenizer,
    pipeline: &PipelineTokenizer,
    pretok: &PipelinePreTokenizer,
    inputs: &[&str],
    repeats: usize,
) {
    let total_bytes: usize = inputs.iter().map(|s| s.len()).sum();
    let av = oracle.get_added_vocabulary();
    let normalizer = oracle.get_normalizer();
    let model = oracle.get_model();

    // Warm up (fill lazy statics, warm caches) outside measurement.
    for input in inputs {
        encode_timed(av, normalizer, model, pretok, input, &mut Stages::default());
    }

    // Reconstructed, per-stage.
    let mut s = Stages::default();
    let mut tokens = 0u64;
    for _ in 0..repeats {
        for input in inputs {
            tokens += encode_timed(av, normalizer, model, pretok, input, &mut s) as u64;
        }
    }

    // Ground truth: the real encode (total time + allocs, not per-stage).
    let a0 = allocs();
    let start = Instant::now();
    let mut real_tokens = 0u64;
    for _ in 0..repeats {
        for input in inputs {
            real_tokens += pipeline.encode(input, false).unwrap().len() as u64;
        }
    }
    let real_wall = start.elapsed();
    let real_allocs = allocs() - a0;

    report(
        name,
        tag,
        inputs.len(),
        &s,
        tokens,
        total_bytes,
        repeats,
        real_wall,
        real_allocs,
        real_tokens,
    );
}

/// Reconstruct `PipelineTokenizer::encode` (+ `tokenize_chunk`) stage-by-stage,
/// timing + counting allocations for each region. Kept structurally 1-1 with
/// the real code: fresh `output`/`pre_tokens` Vecs per call, a single
/// `get_next_special_token` scan loop over the remaining input (special tokens
/// are matched on the raw input only; offsets are discarded), the chunk before
/// each match tokenized via [`tokenize_chunk`], reused `pre_tokens` via
/// `clear()`, and a single `output` extended in place.
fn encode_timed(
    av: &AddedVocabulary,
    normalizer: Option<&NormalizerWrapper>,
    model: &ModelWrapper,
    pretok: &PipelinePreTokenizer,
    input: &str,
    s: &mut Stages,
) -> usize {
    let mut output: Vec<PipelineToken> = Vec::new();
    let mut pre_tokens: Vec<Split> = Vec::new();

    let mut offset = 0usize;
    loop {
        let (a0, t0) = (allocs(), Instant::now());
        let next = av.get_next_special_token(&input[offset..], false);
        s.extract.add(t0.elapsed(), allocs() - a0);

        if let Some(((start, end), token)) = next {
            let chunk = &input[offset..offset + start];
            if !chunk.is_empty() {
                tokenize_chunk(
                    normalizer,
                    model,
                    pretok,
                    chunk,
                    &mut pre_tokens,
                    &mut output,
                    s,
                );
            }
            output.push(PipelineToken { id: token });
            offset += end;
        } else {
            let chunk = &input[offset..];
            if !chunk.is_empty() {
                tokenize_chunk(
                    normalizer,
                    model,
                    pretok,
                    chunk,
                    &mut pre_tokens,
                    &mut output,
                    s,
                );
            }
            break;
        }
    }

    output.len()
}

/// Mirror of `PipelineTokenizer::tokenize_chunk`: build a `NormalizedString`,
/// normalize, pre-tokenize (reusing `pre_tokens`), then `model.tokenize` each
/// pre-token and extend `output` with the ids. Each region is timed +
/// alloc-counted into `s`.
#[allow(clippy::too_many_arguments)]
fn tokenize_chunk(
    normalizer: Option<&NormalizerWrapper>,
    model: &ModelWrapper,
    pretok: &PipelinePreTokenizer,
    chunk: &str,
    pre_tokens: &mut Vec<Split>,
    output: &mut Vec<PipelineToken>,
    s: &mut Stages,
) {
    let (a0, t0) = (allocs(), Instant::now());
    let mut normalized: NormalizedString = chunk.into();
    if let Some(n) = normalizer {
        n.normalize(&mut normalized).unwrap();
    }
    s.normalize.add(t0.elapsed(), allocs() - a0);
    let normalized_chunk = normalized.get();

    let (a0, t0) = (allocs(), Instant::now());
    pre_tokens.clear();
    pretok.pre_tokenize(normalized_chunk, pre_tokens).unwrap();
    s.pretok.add(t0.elapsed(), allocs() - a0);

    let (a0, t0) = (allocs(), Instant::now());
    for pt in pre_tokens.iter() {
        output.extend(
            model
                .tokenize(&normalized_chunk[pt.range()])
                .unwrap()
                .into_iter()
                .map(|Token { id, .. }| PipelineToken { id }),
        );
    }
    s.tokenize.add(t0.elapsed(), allocs() - a0);
}

#[allow(clippy::too_many_arguments)]
fn report(
    name: &str,
    tag: &str,
    n_inputs: usize,
    s: &Stages,
    tokens: u64,
    total_bytes: usize,
    repeats: usize,
    real_wall: Duration,
    real_allocs: u64,
    real_tokens: u64,
) {
    let steps = [
        ("tokenization (model + extend)", s.tokenize),
        ("normalization (+ NormString)", s.normalize),
        ("special-token extract", s.extract),
        ("pre-tokenization", s.pretok),
    ];
    let sum_time: Duration = steps.iter().map(|(_, a)| a.time).sum();
    let sum_allocs: u64 = steps.iter().map(|(_, a)| a.allocs).sum();
    let sum_ns = sum_time.as_nanos().max(1);
    let pct = |a: &Acc| 100.0 * a.time.as_nanos() as f64 / sum_ns as f64;

    println!("######## {name} ########");
    println!(
        "  {:.2} MB x {repeats} passes, {} tokens\n",
        total_bytes as f64 / 1e6,
        tokens
    );
    println!(
        "  {:<34} {:>9}  {:>6}   {:>13}  {:>10}",
        "step", "time(ms)", "time%", "alloc-calls", "allocs/tok"
    );
    for (label, a) in steps {
        println!(
            "  {:<34} {:>9.1}  {:>5.1}%   {:>13}  {:>10.3}",
            label,
            a.time.as_secs_f64() * 1e3,
            100.0 * a.time.as_nanos() as f64 / sum_ns as f64,
            a.allocs,
            a.allocs as f64 / tokens as f64,
        );
    }
    println!(
        "  {:-<34} {:->9} {:->7} {:->14} {:->11}",
        "", "", "", "", ""
    );
    println!(
        "  {:<34} {:>9.1}  {:>6}   {:>13}  {:>10.3}",
        "sum of steps",
        sum_time.as_secs_f64() * 1e3,
        "100%",
        sum_allocs,
        sum_allocs as f64 / tokens as f64,
    );

    // Ground truth from the real encode.
    let real_mb_s = (total_bytes * repeats) as f64 / 1e6 / real_wall.as_secs_f64();
    println!(
        "\n  ground truth (real PipelineTokenizer::encode): {:.1} ms  {:.1} MB/s",
        real_wall.as_secs_f64() * 1e3,
        real_mb_s
    );
    println!(
        "    total alloc-calls {real_allocs} ({:.3}/tok)  |  reconstruction captures {:.0}% of allocs, {:.0}% of time",
        real_allocs as f64 / real_tokens as f64,
        100.0 * sum_allocs as f64 / real_allocs.max(1) as f64,
        100.0 * sum_time.as_nanos() as f64 / real_wall.as_nanos().max(1) as f64,
    );

    // Machine-parseable row (one per regime) for aggregating a matrix run.
    // Fields: tag n_inputs avg mbps | tok% norm% pretok% extract% | total_a/t |
    //         tok_a/t norm_a/t pretok_a/t extract_a/t
    if std::env::var("STAGES_CSV").is_ok() {
        let avg = total_bytes / n_inputs.max(1);
        let apt = |a: &Acc| a.allocs as f64 / tokens.max(1) as f64;
        println!(
            "CSV\t{tag}\t{n_inputs}\t{avg}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            real_mb_s,
            pct(&s.tokenize),
            pct(&s.normalize),
            pct(&s.pretok),
            pct(&s.extract),
            real_allocs as f64 / real_tokens as f64,
            apt(&s.tokenize),
            apt(&s.normalize),
            apt(&s.pretok),
            apt(&s.extract),
        );
    }
}

fn build_pipeline_pretokenizer(oracle: &Tokenizer) -> Result<PipelinePreTokenizer> {
    Ok(match oracle.get_pre_tokenizer() {
        None => PipelinePreTokenizer::None,
        Some(PreTokenizerWrapper::BertPreTokenizer(p)) => PipelinePreTokenizer::Bert(*p),
        Some(PreTokenizerWrapper::Whitespace(p)) => PipelinePreTokenizer::Whitespace(p.clone()),
        Some(other) => {
            return Err(format!("unsupported pre-tokenizer for pipeline: {other:?}").into())
        }
    })
}

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
