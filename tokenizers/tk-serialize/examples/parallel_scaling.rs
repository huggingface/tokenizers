//! Encode throughput vs thread count, one row per (tokenizer, corpus, shape, threads).
//!
//! ```text
//! cargo run --release -p tk-serialize --example parallel_scaling -- \
//!     --tokenizer llama-3 --corpus english --shape single --threads 1,2,4,6
//! ```
//!
//! Flags take comma-separated lists; rows are their cross product. `--csv` for machine output.
//! - shapes: `single` (one document), `batch` (~4KB documents), `mixed` (one large document last,
//!   plus small ones)
//! - corpora: `english`, `multilingual`, `cjk`, `base64`, `code` (this tree's `tk-encode/src`, so
//!   pass a path instead when comparing branches), or a file / directory path
//!
//! Results are drained through the streaming iterator. Each row's ids are checked against the
//! 1-thread row.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tk_encode::parallelism::set_num_threads;
use tk_encode::pipeline::{EncodeOptions, Inputs, PipelineTokenizer};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

const TOKENIZERS: &[(&str, &str)] = &[
    ("llama-3", "llama-3-tokenizer.json"),
    ("roberta", "roberta.json"),
    ("bert", "bert-wiki.json"),
    ("deepseek-v3", "deepseek-v3-tokenizer.json"),
];

const CORPORA: &[&str] = &["english", "multilingual", "cjk", "code", "base64"];
const SHAPES: &[&str] = &["single", "batch", "mixed"];

/// Document size for the `batch` and `mixed` shapes: about a page of text.
const DOC_BYTES: usize = 4 * 1024;

struct Args {
    tokenizers: Vec<String>,
    corpora: Vec<String>,
    shapes: Vec<String>,
    threads: Vec<usize>,
    bytes: usize,
    min_time: Duration,
    csv: bool,
}

fn list(value: &str) -> Vec<String> {
    value.split(',').map(str::to_owned).collect()
}

fn default_threads() -> Vec<usize> {
    let max = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut counts: Vec<usize> = std::iter::successors(Some(1usize), |n| Some(n * 2))
        .take_while(|n| *n < max)
        .collect();
    counts.push(max);
    counts
}

fn parse_args() -> Args {
    let mut args = Args {
        tokenizers: TOKENIZERS
            .iter()
            .map(|(name, _)| (*name).to_owned())
            .collect(),
        corpora: CORPORA.iter().map(|c| (*c).to_owned()).collect(),
        shapes: SHAPES.iter().map(|s| (*s).to_owned()).collect(),
        threads: default_threads(),
        bytes: 4 << 20,
        min_time: Duration::from_secs(2),
        csv: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--tokenizer" => args.tokenizers = list(&value()),
            "--corpus" => args.corpora = list(&value()),
            "--shape" => args.shapes = list(&value()),
            "--threads" => {
                args.threads = list(&value())
                    .iter()
                    .map(|t| t.parse().expect("--threads takes integers"))
                    .collect()
            }
            "--bytes" => args.bytes = value().parse().expect("--bytes takes an integer"),
            "--secs" => {
                args.min_time =
                    Duration::from_secs_f64(value().parse().expect("--secs takes a number"))
            }
            "--csv" => args.csv = true,
            other => panic!("unknown flag {other}"),
        }
    }
    args
}

/// A fixture name from [`TOKENIZERS`], or a path to any `tokenizer.json`.
fn load(name: &str) -> PipelineTokenizer {
    let path = TOKENIZERS.iter().find(|(n, _)| *n == name).map_or_else(
        || PathBuf::from(name),
        |(_, file)| Path::new(DATA).join(file),
    );
    // fixtures are v1.0, the reader wants canonical v2.0
    let canonical =
        tk_convert::canonicalize_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    tk_serialize::from_json(&canonical).unwrap()
}

/// Every `.rs` file under `dir`, concatenated in a stable order.
fn rust_sources(dir: &Path, out: &mut String) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push_str(&std::fs::read_to_string(&path).unwrap());
        }
    }
}

/// Whitespace-free and token-dense: few natural cut points.
fn base64(len: usize) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..len)
        .map(|_| {
            // xorshift64: same bytes on every machine
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ALPHABET[(state >> 58) as usize] as char
        })
        .collect()
}

/// `len` bytes of the named corpus, repeated if the source is shorter, cut at a char boundary.
/// Anything that is not a known name is read as a path: a file, or a directory of `.rs` files.
fn corpus(name: &str, len: usize) -> String {
    let source = match name {
        "english" => std::fs::read_to_string(Path::new(DATA).join("big.txt")).unwrap(),
        "multilingual" => std::fs::read_to_string(Path::new(DATA).join("xnli.txt")).unwrap(),
        "cjk" => std::fs::read_to_string(Path::new(DATA).join("unigram_wagahaiwa_nekodearu.txt"))
            .unwrap(),
        "code" => {
            let mut out = String::new();
            rust_sources(
                &Path::new(env!("CARGO_MANIFEST_DIR")).join("../tk-encode/src"),
                &mut out,
            );
            out
        }
        "base64" => return base64(len),
        path if Path::new(path).is_dir() => {
            let mut out = String::new();
            rust_sources(Path::new(path), &mut out);
            out
        }
        path => std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}")),
    };
    let mut out = String::with_capacity(len + source.len());
    while out.len() < len {
        out.push_str(&source);
    }
    let mut end = len;
    while !out.is_char_boundary(end) {
        end -= 1;
    }
    out.truncate(end);
    out
}

/// Cut `text` into documents of about `doc` bytes, at char boundaries.
fn documents(text: &str, doc: usize) -> Vec<String> {
    let mut docs = Vec::with_capacity(text.len() / doc + 1);
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + doc).min(text.len());
        while !text.is_char_boundary(end) {
            end += 1;
        }
        docs.push(text[start..end].to_owned());
        start = end;
    }
    docs
}

fn inputs(shape: &str, text: &str) -> Vec<String> {
    match shape {
        "single" => vec![text.to_owned()],
        "batch" => documents(text, DOC_BYTES),
        "mixed" => {
            let mut half = text.len() / 2;
            while !text.is_char_boundary(half) {
                half += 1;
            }
            // big document last: worst case for in-order scheduling
            let mut docs = documents(&text[half..], DOC_BYTES);
            docs.push(text[..half].to_owned());
            docs
        }
        other => panic!("unknown shape {other}"),
    }
}

/// Token count and an order-sensitive id checksum per input, folded in input order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Digest {
    tokens: usize,
    checksum: u64,
}

/// Untimed encode + id checksum (a serial pass that would skew high thread counts).
fn digest(tokenizer: &PipelineTokenizer, docs: &[String], options: &EncodeOptions) -> Digest {
    let mut per_input = vec![(0usize, 0u64); docs.len()];
    for (i, result) in tokenizer.encode(Inputs::from(docs.to_vec()), options) {
        let encoding = result.unwrap();
        let checksum = encoding
            .ids()
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, id| {
                (h ^ u64::from(id.id())).wrapping_mul(0x100_0000_01b3)
            });
        per_input[i] = (encoding.len(), checksum);
    }
    per_input.iter().fold(
        Digest {
            tokens: 0,
            checksum: 0,
        },
        |d, &(n, c)| Digest {
            tokens: d.tokens + n,
            checksum: d.checksum.rotate_left(5) ^ c,
        },
    )
}

/// Timed encode; the input copy happens before the clock starts.
fn timed(tokenizer: &PipelineTokenizer, docs: &[String], options: &EncodeOptions) -> Duration {
    let inputs = Inputs::from(docs.to_vec());
    let start = Instant::now();
    for (_, result) in tokenizer.encode(inputs, options) {
        black_box(result.unwrap());
    }
    start.elapsed()
}

fn main() {
    let args = parse_args();
    let options = EncodeOptions::default();
    if args.csv {
        println!("tokenizer,corpus,shape,threads,bytes,mb_per_s,speedup,tokens,iters");
    } else {
        println!(
            "{:<12} {:<13} {:<7} {:>7} {:>10} {:>8} {:>11}",
            "tokenizer", "corpus", "shape", "threads", "MB/s", "speedup", "tokens"
        );
    }
    for name in &args.tokenizers {
        let tokenizer = load(name);
        for corpus_name in &args.corpora {
            let text = corpus(corpus_name, args.bytes);
            for shape in &args.shapes {
                let docs = inputs(shape, &text);
                let bytes: usize = docs.iter().map(String::len).sum();
                let mut baseline: Option<(f64, Digest)> = None;
                for &threads in &args.threads {
                    set_num_threads(threads);
                    // also the warm-up: pool rebuild + per-thread scratch
                    let digest = digest(&tokenizer, &docs, &options);
                    let mut times = Vec::new();
                    let start = Instant::now();
                    while times.len() < 3 || start.elapsed() < args.min_time {
                        times.push(timed(&tokenizer, &docs, &options));
                    }
                    times.sort();
                    let median = times[times.len() / 2].as_secs_f64();
                    let mbps = bytes as f64 / median / 1e6;
                    let base = *baseline.get_or_insert((mbps, digest));
                    assert_eq!(
                        digest, base.1,
                        "{name}/{corpus_name}/{shape}: {threads} threads changed the ids"
                    );
                    let speedup = mbps / base.0;
                    if args.csv {
                        println!(
                            "{name},{corpus_name},{shape},{threads},{bytes},{mbps:.1},{speedup:.2},{},{}",
                            digest.tokens,
                            times.len()
                        );
                    } else {
                        println!(
                            "{name:<12} {corpus_name:<13} {shape:<7} {threads:>7} {mbps:>10.1} {speedup:>7.2}x {:>11}",
                            digest.tokens
                        );
                    }
                }
            }
        }
    }
    set_num_threads(0);
}
