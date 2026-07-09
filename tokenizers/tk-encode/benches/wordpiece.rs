//! WordPiece model micro-benchmark — decomposes the win into its independent levers so we don't
//! misattribute it. All four variants produce identical ids (asserted byte-exact); they differ only
//! in how the per-position longest-match is found:
//!
//!   V1  legacy       : AHashMap, UNBOUNDED greedy longest-first + early-exit (the shipped baseline)
//!   V2  bounded map  : AHashMap, bounded by longest vocab token           (isolates the *bounding* win)
//!   V3  bounded mphf : BucketVocabStore.get_bytes, bounded, scalar        (isolates AHashMap -> MPHF)
//!   V4  streamed mphf: BucketVocabStore.longest_prefix_match (index_stream, prefetch, no early-exit)
//!                                                                          (isolates *streaming*)
//!
//! V3 is what `PipelineWordPiece` ships. NOTE: on aarch64 ptr_hash's prefetch is a no-op, so V4 only
//! shows its instruction-level-parallelism here; its prefetch win is x86/x86_64-only.
//!
//! Self-contained (no fixtures): single chars are always in-vocab (guarantees a fallback, no unk), plus
//! a deterministic ~40% subset of 2/3/4-letter combos (plain and `##`) so greedy does real multi-probe
//! work per position — not the degenerate "every longest probe hits" case.

use ahash::AHashMap;
use std::hint::black_box;
use std::time::Instant;
use tk_encode::added_vocabulary::bucket_vocab_store::BucketVocabStore;
use tk_encode::models::wordpiece::{PipelineWordPiece, WordPiece};
use tk_encode::pipeline::Model as PipelineModel;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn build_vocab() -> WordPiece {
    let mut vocab: AHashMap<String, u32> = AHashMap::new();
    let mut next = 0u32;
    let ins = |s: String, v: &mut AHashMap<String, u32>, n: &mut u32| {
        v.entry(s).or_insert_with(|| {
            let id = *n;
            *n += 1;
            id
        });
    };
    ins("[UNK]".to_string(), &mut vocab, &mut next);
    // Single chars always present -> greedy can always fall back to 1 char, so no word is unk.
    for a in b'a'..=b'z' {
        let c = (a as char).to_string();
        ins(c.clone(), &mut vocab, &mut next);
        ins(format!("##{c}"), &mut vocab, &mut next);
    }
    // ~40% of 2/3/4-letter combos, deterministically chosen, both plain and `##`.
    let mut rng = Lcg(0xDEAD_BEEF_1234_5678);
    let maybe = |s: String, v: &mut AHashMap<String, u32>, n: &mut u32, r: &mut Lcg| {
        if r.next_u64() % 5 < 2 {
            ins(s.clone(), v, n);
            ins(format!("##{s}"), v, n);
        }
    };
    for a in b'a'..=b'z' {
        for b in b'a'..=b'z' {
            maybe(
                format!("{}{}", a as char, b as char),
                &mut vocab,
                &mut next,
                &mut rng,
            );
            for c in b'a'..=b'z' {
                maybe(
                    format!("{}{}{}", a as char, b as char, c as char),
                    &mut vocab,
                    &mut next,
                    &mut rng,
                );
            }
        }
    }
    // A tail of longer tokens (len 4..=12) so `max_token_len` is realistic (~BERT), not just 3 — bounding
    // then can't trivially collapse every position to one probe.
    for _ in 0..4000 {
        let len = 4 + (rng.next_u64() % 9) as usize;
        let s: String = (0..len)
            .map(|_| (b'a' + (rng.next_u64() % 26) as u8) as char)
            .collect();
        ins(s.clone(), &mut vocab, &mut next);
        ins(format!("##{s}"), &mut vocab, &mut next);
    }
    WordPiece::builder().vocab(vocab).build().unwrap()
}

fn corpus(n_words: usize) -> Vec<String> {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    (0..n_words)
        .map(|_| {
            let len = 4 + (rng.next_u64() % 17) as usize; // 4..=20 chars
            (0..len)
                .map(|_| (b'a' + (rng.next_u64() % 26) as u8) as char)
                .collect()
        })
        .collect()
}

/// Bounded greedy longest-first + early-exit; `lookup(candidate_bytes) -> id`. Shared by V2/V3.
fn greedy_scalar<F: Fn(&[u8]) -> Option<u32>>(
    word: &str,
    prefix: &[u8],
    max_token_len: usize,
    unk: u32,
    out: &mut Vec<u32>,
    lookup: F,
) {
    let bytes = word.as_bytes();
    let mut buf = Vec::with_capacity(max_token_len);
    let mut ends: Vec<usize> = Vec::with_capacity(max_token_len);
    let mut start = 0usize;
    let base = out.len();
    while start < word.len() {
        let prefix_len = if start > 0 { prefix.len() } else { 0 };
        let cap_word = max_token_len.saturating_sub(prefix_len);
        let mut capped = 0usize;
        ends.clear();
        for (idx, ch) in word[start..].char_indices() {
            let e = idx + ch.len_utf8();
            if e > cap_word {
                break;
            }
            capped = e;
            ends.push(e);
        }
        buf.clear();
        buf.extend_from_slice(&prefix[..prefix_len]);
        buf.extend_from_slice(&bytes[start..start + capped]);
        let mut hit = None;
        for &e in ends.iter().rev() {
            if let Some(id) = lookup(&buf[..prefix_len + e]) {
                hit = Some((e, id));
                break;
            }
        }
        match hit {
            Some((e, id)) => {
                out.push(id);
                start += e;
            }
            None => {
                out.truncate(base);
                out.push(unk);
                return;
            }
        }
    }
}

/// V4: bounded, but the length probes go through `index_stream` (prefetch, no early exit).
fn greedy_streamed(
    word: &str,
    prefix: &[u8],
    max_token_len: usize,
    unk: u32,
    store: &BucketVocabStore,
    out: &mut Vec<u32>,
) {
    let bytes = word.as_bytes();
    let mut buf = Vec::with_capacity(max_token_len);
    let mut ends: Vec<usize> = Vec::with_capacity(max_token_len);
    let mut start = 0usize;
    let base = out.len();
    while start < word.len() {
        let prefix_len = if start > 0 { prefix.len() } else { 0 };
        let cap_word = max_token_len.saturating_sub(prefix_len);
        ends.clear();
        let mut capped = 0usize;
        for (idx, ch) in word[start..].char_indices() {
            let e = idx + ch.len_utf8();
            if e > cap_word {
                break;
            }
            capped = e;
            ends.push(prefix_len + e);
        }
        buf.clear();
        buf.extend_from_slice(&prefix[..prefix_len]);
        buf.extend_from_slice(&bytes[start..start + capped]);
        match store.longest_prefix_match(&buf, &ends) {
            Some((matched_len, id)) => {
                out.push(id);
                start += matched_len - prefix_len;
            }
            None => {
                out.truncate(base);
                out.push(unk);
                return;
            }
        }
    }
}

fn time<F: FnMut(&str, &mut Vec<u32>)>(
    words: &[String],
    bytes: usize,
    iters: u32,
    mut run: F,
) -> (f64, u64) {
    let mut out = Vec::with_capacity(64);
    let mut sink = 0u64;
    let t = Instant::now();
    for _ in 0..iters {
        for w in words {
            out.clear();
            run(w, &mut out);
            sink = sink.wrapping_add(out.len() as u64);
        }
    }
    (
        t.elapsed().as_nanos() as f64 / (iters as usize * bytes) as f64,
        sink,
    )
}

fn main() {
    let wp = build_vocab();
    let prefix = wp.continuing_subword_prefix.as_bytes().to_vec();
    let unk = *wp.vocab.get("[UNK]").unwrap();
    let max_token_len = wp.vocab.keys().map(|k| k.len()).max().unwrap();
    let store = BucketVocabStore::build(
        wp.vocab
            .iter()
            .map(|(s, &id)| (s.as_bytes().to_vec(), id))
            .collect(),
    );
    let pw = PipelineWordPiece::from_wordpiece(&wp);
    let words = corpus(5000);
    let total: usize = words.iter().map(|w| w.len()).sum();

    // Byte-exactness of every variant against the shipped legacy path.
    let (mut base, mut v2, mut v3, mut v4, mut vp, mut tmp) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut ok = true;
    for w in &words {
        tmp.clear();
        PipelineModel::tokenize_pipeline(&wp, w, &mut tmp).unwrap();
        base.clear();
        base.extend(tmp.iter().map(|t| t.id));

        v2.clear();
        greedy_scalar(w, &prefix, max_token_len, unk, &mut v2, |q| {
            wp.vocab.get(std::str::from_utf8(q).unwrap()).copied()
        });
        v3.clear();
        greedy_scalar(w, &prefix, max_token_len, unk, &mut v3, |q| {
            store.get_bytes(q)
        });
        v4.clear();
        greedy_streamed(w, &prefix, max_token_len, unk, &store, &mut v4);
        tmp.clear();
        PipelineModel::tokenize_pipeline(&pw, w, &mut tmp).unwrap();
        vp.clear();
        vp.extend(tmp.iter().map(|t| t.id));

        if v2 != base || v3 != base || v4 != base || vp != base {
            ok = false;
            eprintln!("DIVERGE on {w:?}: base={base:?} v2={v2:?} v3={v3:?} v4={v4:?} vp={vp:?}");
            break;
        }
    }

    let iters = 40;
    // warm
    let _ = time(&words, total, 2, |w, _o| {
        let mut t = Vec::new();
        PipelineModel::tokenize_pipeline(&wp, w, &mut t).unwrap();
    });
    let (v1, s1) = time(&words, total, iters, |w, o| {
        let mut t = Vec::new();
        PipelineModel::tokenize_pipeline(&wp, w, &mut t).unwrap();
        o.extend(t.iter().map(|x| x.id));
    });
    let (v2, s2) = time(&words, total, iters, |w, o| {
        greedy_scalar(w, &prefix, max_token_len, unk, o, |q| {
            wp.vocab.get(std::str::from_utf8(q).unwrap()).copied()
        })
    });
    let (v3, s3) = time(&words, total, iters, |w, o| {
        greedy_scalar(w, &prefix, max_token_len, unk, o, |q| store.get_bytes(q))
    });
    let (v4, s4) = time(&words, total, iters, |w, o| {
        greedy_streamed(w, &prefix, max_token_len, unk, &store, o)
    });
    let (vp, s5) = time(&words, total, iters, |w, o| {
        let mut t = Vec::new();
        PipelineModel::tokenize_pipeline(&pw, w, &mut t).unwrap();
        o.extend(t.iter().map(|x| x.id));
    });
    black_box((s1, s2, s3, s4, s5));

    println!("\nWordPiece model lookup — {} vocab tokens, longest {max_token_len}B, {} words, {total} bytes", wp.vocab.len(), words.len());
    println!("  V1 legacy    (AHashMap, unbounded)      : {v1:>7.3} ns/byte  (1.00x)");
    println!("  V2 bounded   (AHashMap, bounded)        : {v2:>7.3} ns/byte  ({:.2}x)   isolates bounding", v1 / v2);
    println!("  V3 mphf      (BucketVocabStore, scalar) : {v3:>7.3} ns/byte  ({:.2}x)   isolates AHashMap->MPHF", v1 / v3);
    println!("  V4 streamed  (index_stream, prefetch)   : {v4:>7.3} ns/byte  ({:.2}x)   isolates streaming (x86 only)", v1 / v4);
    println!("  ** PipelineWordPiece (shipped, MPHF+bounded+reused bufs): {vp:>7.3} ns/byte  ({:.2}x) **", v1 / vp);
    println!(
        "  byte-exact across all variants: {}",
        if ok { "✓" } else { "✗ MISMATCH" }
    );
    assert!(ok, "a variant diverged from legacy");
}
