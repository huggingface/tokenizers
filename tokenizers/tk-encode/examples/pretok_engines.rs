//! How fast is `classify + fsm` against a real regex engine?
//!
//! Most tokenizers pre-tokenize by running a regex over the text. This crate instead classifies every
//! character into an "atom" class in one SIMD pass and then cuts with a small state machine
//! (`atomsplit`), which produces the same pieces without a regex engine. This binary puts the two
//! side by side: for every model in the manifest it takes that model's own pre-tokenizer regex and
//! times it under three engines — oniguruma and PCRE2 (both C, PCRE2 JIT-compiled) and fancy-regex
//! (pure Rust) — plus a logos DFA lexer where the grammar can be expressed, over the same corpora
//! `fixture_bench` uses.
//!
//! It also reports the classify pass alone, SIMD and scalar, so the comparison can be read both with
//! and without SIMD: the state machine is the same scalar jump table either way.
//!
//! This lives apart from `fixture_bench` because the engines are extra dependencies, two of them C
//! libraries, and one of them is the `fancy-regex` backend itself. Pulling them into the throughput
//! benchmark would mean benchmarking a build nobody ships. Built with `--features bench-engines`.
//!
//! Emits `{model: {fixture: {cls_simd, cls_scalar, onig, fancy, pcre2, logos}}}` on stdout, which CI
//! merges into the benchmark report as each row's `pretok_vs_regex`.

mod bench_common;

use bench_common::{load_fixtures, model_path, pretok_regexes, shard, timed_ns};
use logos::Logos;
use serde_json::{Map, Value, json};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let manifest = shard(&args);
    let fixtures = load_fixtures();

    let mut models = Map::new();
    for entry in &manifest {
        let name = entry["name"].as_str().unwrap().to_string();
        let regexes = pretok_regexes(&model_path(entry));
        eprintln!("== {name} ({} regex(es)) ==", regexes.len());
        let mut rows = Map::new();
        for f in &fixtures {
            let corpus: String = f.chunks.concat();
            let cls_simd = classify_ns(corpus.as_bytes(), false);
            let cls_scalar = classify_ns(corpus.as_bytes(), true);
            let onig = regex_reference_ns::<onig::Regex>(&corpus, &regexes);
            let fancy = regex_reference_ns::<fancy_regex::Regex>(&corpus, &regexes);
            let pcre2 = regex_reference_ns::<pcre2::bytes::Regex>(&corpus, &regexes);
            let logos = logos_reference_ns(&regexes, &corpus);
            eprintln!(
                "  {} cls {cls_simd:.2}/{cls_scalar:.2} · onig {} · fancy {} · pcre2 {} · logos {}",
                f.name,
                ns_or_dash(onig),
                ns_or_dash(fancy),
                ns_or_dash(pcre2),
                ns_or_dash(logos),
            );
            rows.insert(
                f.name.clone(),
                json!({
                    "cls_simd": cls_simd,
                    "cls_scalar": cls_scalar,
                    "onig": onig,
                    "fancy": fancy,
                    "pcre2": pcre2,
                    "logos": logos,
                }),
            );
        }
        models.insert(name, Value::Object(rows));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Object(models)).unwrap()
    );
}

/// ns/byte for the log line, or `—` for a model this engine has no number for.
fn ns_or_dash(v: Option<f64>) -> String {
    v.map_or("—".into(), |v| format!("{v:.2}"))
}

/// Median ns/byte to classify `bytes` once via the SIMD or scalar path.
fn classify_ns(bytes: &[u8], scalar: bool) -> f64 {
    let mut tags = vec![0u8; bytes.len()];
    timed_ns(bytes.len(), || {
        if scalar {
            atomsplit::classify::classify_scalar(bytes, &mut tags);
        } else {
            atomsplit::classify::classify(bytes, &mut tags);
        }
        tags[bytes.len() / 2] as usize
    })
}

// ── regex-engine references ─────────────────────────────────────────────────
// The pipeline's `pre_tokenize` stage is `classify (SIMD) + fsm`; these reference
// numbers time the model's own pre-tokenizer regex(es) — the split a regex-based
// tokenizer actually pays for — under three real engines. Each engine only needs to
// enumerate matches; the composed Isolated split chain is shared.

/// A regex engine timed through the composed split chain.
trait SplitEngine: Sized {
    fn compile(pattern: &str) -> Option<Self>;
    /// Call `on_match(start, end)` for every match in `hay`, in order.
    fn for_each_match(&self, hay: &str, on_match: impl FnMut(usize, usize));
}

/// Oniguruma (C) — what the reference tokenizer itself uses.
impl SplitEngine for onig::Regex {
    fn compile(pattern: &str) -> Option<Self> {
        onig::Regex::new(pattern).ok()
    }
    fn for_each_match(&self, hay: &str, mut on_match: impl FnMut(usize, usize)) {
        for (s, e) in self.find_iter(hay) {
            on_match(s, e);
        }
    }
}

/// fancy-regex (pure Rust). `find_iter` yields `Result<Match, _>`; a match error
/// aborts that piece's pass (rare, backtrack-limit) and it is left un-split.
impl SplitEngine for fancy_regex::Regex {
    fn compile(pattern: &str) -> Option<Self> {
        fancy_regex::Regex::new(pattern).ok()
    }
    fn for_each_match(&self, hay: &str, mut on_match: impl FnMut(usize, usize)) {
        for m in self.find_iter(hay) {
            let Ok(m) = m else { break };
            on_match(m.start(), m.end());
        }
    }
}

/// PCRE2 (C) — built with `utf(true).ucp(true)` so `\p{L}`/`\p{N}`/`\s` are
/// Unicode-aware and byte offsets land on char boundaries, matching the other
/// engines, and **JIT-compiled** so PCRE2 is benched at its best.
impl SplitEngine for pcre2::bytes::Regex {
    fn compile(pattern: &str) -> Option<Self> {
        pcre2::bytes::RegexBuilder::new()
            .utf(true)
            .ucp(true)
            .jit_if_available(true)
            .build(pattern)
            .ok()
    }
    fn for_each_match(&self, hay: &str, mut on_match: impl FnMut(usize, usize)) {
        for m in self.find_iter(hay.as_bytes()) {
            let Ok(m) = m else { break };
            on_match(m.start(), m.end());
        }
    }
}

/// ns/byte for the composed Isolated split chain under engine `E` — each regex splits
/// the previous pieces (gaps + matches), exactly how the reference tokenizer applies a
/// `Sequence` of Splits. `None` when the model has no regex pre-tokenizer, or the
/// engine rejects a pattern.
fn regex_reference_ns<E: SplitEngine>(text: &str, patterns: &[String]) -> Option<f64> {
    if patterns.is_empty() || text.is_empty() {
        return None;
    }
    let engines: Vec<E> = patterns
        .iter()
        .map(|p| E::compile(p))
        .collect::<Option<_>>()?;
    Some(timed_ns(text.len(), || {
        let mut pieces = vec![(0usize, text.len())];
        for re in &engines {
            let mut next = Vec::with_capacity(pieces.len() * 2);
            for (s, e) in pieces.drain(..) {
                let mut prev = 0usize;
                re.for_each_match(&text[s..e], |ms, me| {
                    if ms > prev {
                        next.push((s + prev, s + ms));
                    }
                    next.push((s + ms, s + me));
                    prev = me;
                });
                if prev < e - s {
                    next.push((s + prev, e));
                }
            }
            pieces = next;
        }
        pieces.len()
    }))
}

// logos DFA lexers approximating the GPT splits (no look-ahead / case-insensitive →
// boundaries differ slightly; a raw-throughput reference like fancy, not a byte-exact
// oracle). Only families logos can express get a number; deepseek / variants /
// non-regex pretoks report null.
#[derive(Logos)]
enum LGpt2 {
    #[regex(r"'s|'t|'re|'ve|'m|'ll|'d")]
    Contraction,
    #[regex(r" ?\p{L}+")]
    Word,
    #[regex(r" ?\p{N}+")]
    Num,
    #[regex(r" ?[^\s\p{L}\p{N}]+")]
    Other,
    #[regex(r"\s+")]
    Space,
}
#[derive(Logos)]
enum LCl100k {
    #[regex(r"'s|'t|'re|'ve|'m|'ll|'d", priority = 5)]
    Contraction,
    #[regex(r"[^\r\n\p{L}\p{N}]?\p{L}+", priority = 4)]
    Word,
    #[regex(r"\p{N}\p{N}?\p{N}?")]
    Num,
    #[regex(r" ?[^\s\p{L}\p{N}]+[\r\n]*", priority = 2)]
    Other,
    #[regex(r"\s+")]
    Space,
}
#[derive(Logos)]
enum LO200k {
    #[regex(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+('s|'t|'re|'ve|'m|'ll|'d)?", priority = 6)]
    LettersA,
    #[regex(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*('s|'t|'re|'ve|'m|'ll|'d)?", priority = 5)]
    LettersB,
    #[regex(r"\p{N}\p{N}?\p{N}?")]
    Num,
    #[regex(r" ?[^\s\p{L}\p{N}]+[\r\n/]*", priority = 2)]
    Other,
    #[regex(r"\s+")]
    Space,
}

fn lex_count<'s, T: Logos<'s, Source = str>>(s: &'s str) -> usize
where
    T::Extras: Default,
{
    let mut lex = T::lexer(s);
    let mut n = 0;
    while lex.next().is_some() {
        n += 1;
    }
    n
}

/// logos throughput (ns/byte) when the model's pre-tokenizer is a single regex logos can
/// express (matched against the canonical gpt2/cl100k/o200k specs); `None` otherwise.
fn logos_reference_ns(regexes: &[String], text: &str) -> Option<f64> {
    if text.is_empty() || regexes.len() != 1 {
        return None;
    }
    let r = regexes[0].as_str();
    let f: fn(&str) -> usize = if r == atomsplit::regexes::GPT2 {
        |s| lex_count::<LGpt2>(s)
    } else if r == atomsplit::regexes::CL100K {
        |s| lex_count::<LCl100k>(s)
    } else if r == atomsplit::regexes::O200K {
        |s| lex_count::<LO200k>(s)
    } else {
        return None;
    };
    Some(timed_ns(text.len(), || f(text)))
}
