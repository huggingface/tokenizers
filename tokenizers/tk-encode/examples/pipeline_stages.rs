//! Per-stage timing + allocation breakdown of `PipelineTokenizer::encode`.
//!
//! `encode` runs its stages inline, so to attribute time (and heap allocations)
//! to each step we reconstruct the exact same stages here from the tokenizer's
//! public components and wrap each region in a timer + an allocation counter (a
//! counting global allocator). The real `PipelineTokenizer::encode` is also run
//! as a ground-truth cross-check that the reconstruction captures ~all the work.
//!
//! The current `encode` drives a `SpecialSegmentIterator` at two sites:
//!   * `encode` iterates it over the **raw** input (`normalized = false`),
//!     handing each text `Segment` to `tokenize_chunk` and pushing special
//!     tokens straight to `output`.
//!   * `tokenize_chunk` normalizes its chunk to a `Cow<str>` (no
//!     `NormalizedString`), then iterates a second `SpecialSegmentIterator` over
//!     the **normalized** text (`normalized = true`), pre-tokenizing + running
//!     the model on each text `Segment`.
//!
//! We mirror that with [`for_each_segment`], a driver reproducing
//! `SpecialSegmentIterator::next` so each call site reads like the real
//! `match segment { .. }`. The `get_next_special_token` calls at both sites are
//! tallied under one "special-token extract" bucket. Keep in sync with `encode`.
//!
//! Usage:
//!   cargo run --release --example pipeline_stages -- [tokenizer.json] [text_file] [repeats]

use std::alloc::{GlobalAlloc, Layout, System};
use std::borrow::Cow;
use std::convert::TryFrom;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use tk_encode::pipeline::{
    Normalizer as PipeNormalizer, PipelinePatternMatcher, PipelinePreTokenizer, PipelineToken,
    PipelineTokenizer, PreTokenizer as PipePreTok, Segment, Split,
};
use tk_encode::{
    AddedVocabulary, Model, ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, Token, Tokenizer,
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

/// Reproduce `SpecialSegmentIterator::next`: scan `input` for the next special
/// token (timing that scan into `s.extract`), yielding the text before it and
/// the token itself as [`Segment`]s to `on_segment`, then the trailing text.
/// This is the shared driver both `encode` (raw input, `normalized = false`) and
/// `tokenize_chunk` (normalized text, `normalized = true`) iterate over.
fn for_each_segment<'a>(
    av: &AddedVocabulary,
    input: &'a str,
    normalized: bool,
    s: &mut Stages,
    mut on_segment: impl FnMut(Segment<'a>, &mut Stages),
) {
    let mut offset = 0usize;
    loop {
        let remaining = &input[offset..];
        if remaining.is_empty() {
            break;
        }
        let (a0, t0) = (allocs(), Instant::now());
        let next = av.get_next_special_token(remaining, normalized);
        s.extract.add(t0.elapsed(), allocs() - a0);

        match next {
            Some(((start, end), token)) => {
                let before = &input[offset..offset + start];
                if !before.is_empty() {
                    on_segment(Segment::Text(before), s);
                }
                on_segment(Segment::SpecialToken(token), s);
                offset += end;
            }
            None => {
                on_segment(Segment::Text(remaining), s);
                break;
            }
        }
    }
}

/// Mirror of `PipelineTokenizer::encode`: fresh `output`/`pre_tokens` Vecs, then
/// iterate the outer `SpecialSegmentIterator` over the **raw** input — text
/// segments go to [`tokenize_chunk`], special tokens straight to `output`.
fn encode_timed(
    av: &AddedVocabulary,
    normalizer: Option<&NormalizerWrapper>,
    model: &ModelWrapper,
    pretok: &PipelinePreTokenizer,
    input: &str,
    s: &mut Stages,
) -> usize {
    let mut output: Vec<PipelineToken> = Vec::with_capacity(1.max(input.len() / 2));
    let mut pre_tokens: Vec<Split> = Vec::with_capacity(4096);

    for_each_segment(av, input, false, s, |segment, s| match segment {
        Segment::SpecialToken(token) => output.push(PipelineToken { id: token }),
        Segment::Text(chunk) => tokenize_chunk(
            av,
            normalizer,
            model,
            pretok,
            chunk,
            &mut pre_tokens,
            &mut output,
            s,
        ),
    });

    output.len()
}

/// Mirror of `PipelineTokenizer::tokenize_chunk`: normalize `chunk` to a
/// `Cow<str>` (the allocation-light path — no `NormalizedString`), then iterate
/// the inner `SpecialSegmentIterator` over the **normalized** text. Text
/// segments go through [`tokenize_text`] (pre-tokenize + `model.tokenize`);
/// special tokens are pushed straight to `output`. Normalization is timed into
/// `s.normalize`; the inner scan shares the `s.extract` bucket with the outer.
#[allow(clippy::too_many_arguments)]
fn tokenize_chunk(
    av: &AddedVocabulary,
    normalizer: Option<&NormalizerWrapper>,
    model: &ModelWrapper,
    pretok: &PipelinePreTokenizer,
    chunk: &str,
    pre_tokens: &mut Vec<Split>,
    output: &mut Vec<PipelineToken>,
    s: &mut Stages,
) {
    let (a0, t0) = (allocs(), Instant::now());
    let norm: Cow<str> = match normalizer {
        Some(n) => n.normalize(chunk),
        None => Cow::Borrowed(chunk),
    };
    s.normalize.add(t0.elapsed(), allocs() - a0);
    let normalized: &str = norm.as_ref();

    for_each_segment(av, normalized, true, s, |segment, s| match segment {
        Segment::SpecialToken(token) => output.push(PipelineToken { id: token }),
        Segment::Text(text) => tokenize_text(model, pretok, text, pre_tokens, output, s),
    });
}

/// Pre-tokenize `text` (reusing `pre_tokens` via `clear()`) then `model.tokenize`
/// each pre-token, extending `output` with the ids. Pre-tokenization and the
/// model step are timed + alloc-counted into `s`.
fn tokenize_text(
    model: &ModelWrapper,
    pretok: &PipelinePreTokenizer,
    text: &str,
    pre_tokens: &mut Vec<Split>,
    output: &mut Vec<PipelineToken>,
    s: &mut Stages,
) {
    let (a0, t0) = (allocs(), Instant::now());
    pre_tokens.clear();
    pretok.pre_tokenize(text, pre_tokens).unwrap();
    s.pretok.add(t0.elapsed(), allocs() - a0);

    let (a0, t0) = (allocs(), Instant::now());
    for pt in pre_tokens.iter() {
        output.extend(
            model
                .tokenize(&text[pt.range()])
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
        ("normalization (Cow)", s.normalize),
        ("special-token extract (raw+norm)", s.extract),
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
