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
//! V5 is what `PipelineWordPiece` ships (probe the distinct token lengths, not char-by-char). NOTE: on
//! aarch64 ptr_hash's prefetch is a no-op, so V4 only shows its instruction-level-parallelism here; its
//! prefetch win is x86/x86_64-only.
//!
//! Self-contained (no fixtures): single chars are always in-vocab (guarantees a fallback, no unk), plus
//! a deterministic ~40% subset of 2/3/4-letter combos (plain and `##`) so greedy does real multi-probe
//! work per position — not the degenerate "every longest probe hits" case.

use std::hint::black_box;
use std::time::Instant;
use tk_encode::added_vocabulary::bucket_vocab_store::BucketVocabStore;
use tk_encode::models::wordpiece::{PipelineWordPiece, WordPiece};
use tk_encode::pipeline::Model as PipelineModel;

fn load_vocab(path: &str) -> WordPiece {
    WordPiece::from_file(path).build().unwrap()
}

/// BERT basic pre-tokenization of a real text file: lowercase, split on whitespace, and isolate each
/// punctuation char as its own pre-token — the per-word inputs the WordPiece model actually sees.
fn load_corpus(path: &str, max_words: usize) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut words = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            cur.extend(ch.to_lowercase());
        } else {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            if !ch.is_whitespace() {
                words.push(ch.to_string());
            }
        }
        if words.len() >= max_words {
            break;
        }
    }
    if !cur.is_empty() && words.len() < max_words {
        words.push(cur);
    }
    words
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

/// V5 — "added-tokens" style: probe only the DISTINCT vocab token byte-lengths (descending, first hit
/// wins) instead of shrinking one char at a time. `lens_desc` is the sorted-desc distinct token lengths.
/// Byte-exact: a byte length that splits a codepoint can't equal any valid-UTF-8 token, so the longest
/// hit is identical to the legacy greedy choice.
fn greedy_lenset(
    word: &str,
    prefix: &[u8],
    lens_desc: &[u16],
    unk: u32,
    store: &BucketVocabStore,
    out: &mut Vec<u32>,
) {
    let bytes = word.as_bytes();
    let maxlen = lens_desc.first().copied().unwrap_or(0) as usize;
    let mut buf = Vec::with_capacity(maxlen);
    let mut start = 0usize;
    let base = out.len();
    while start < word.len() {
        let prefix_len = if start > 0 { prefix.len() } else { 0 };
        let remaining = word.len() - start;
        let cap_wp = maxlen.saturating_sub(prefix_len).min(remaining);
        buf.clear();
        buf.extend_from_slice(&prefix[..prefix_len]);
        buf.extend_from_slice(&bytes[start..start + cap_wp]);
        let mut hit = None;
        for &l in lens_desc {
            let l = l as usize;
            if l < prefix_len + 1 {
                break; // descending: no remaining length can hold a word char
            }
            let wp = l - prefix_len;
            if wp > cap_wp {
                continue; // candidate longer than what's left
            }
            if let Some(id) = store.get_bytes(&buf[..prefix_len + wp]) {
                hit = Some((wp, id));
                break;
            }
        }
        match hit {
            Some((wp, id)) => {
                out.push(id);
                start += wp;
            }
            None => {
                out.truncate(base);
                out.push(unk);
                return;
            }
        }
    }
}

/// V6 — length set indexed by the candidate's first content byte (the intended design): at each position
/// probe only the lengths of tokens that actually start with `word[start]`, not every length.
/// `initial[b]`/`cont[b]` are the sorted-desc distinct token byte-lengths for first byte `b` (word-initial
/// vs `##`-continuation). This is what `PipelineWordPiece` ships.
#[allow(clippy::too_many_arguments)]
fn greedy_lenset_fb(
    word: &str,
    prefix: &[u8],
    initial: &[Vec<u16>],
    cont: &[Vec<u16>],
    unk: u32,
    store: &BucketVocabStore,
    out: &mut Vec<u32>,
) {
    let bytes = word.as_bytes();
    let mut buf = Vec::with_capacity(64);
    let mut start = 0usize;
    let base = out.len();
    while start < word.len() {
        let prefix_len = if start > 0 { prefix.len() } else { 0 };
        let lens = if start > 0 {
            &cont[bytes[start] as usize]
        } else {
            &initial[bytes[start] as usize]
        };
        let remaining = word.len() - start;
        let mut hit = None;
        if let Some(&maxl) = lens.first() {
            let cap = (maxl as usize).saturating_sub(prefix_len).min(remaining);
            buf.clear();
            buf.extend_from_slice(&prefix[..prefix_len]);
            buf.extend_from_slice(&bytes[start..start + cap]);
            for &l in lens {
                let l = l as usize;
                if l < prefix_len + 1 {
                    break;
                }
                let wp = l - prefix_len;
                if wp > cap {
                    continue;
                }
                if let Some(id) = store.get_bytes(&buf[..prefix_len + wp]) {
                    hit = Some((wp, id));
                    break;
                }
            }
        }
        match hit {
            Some((wp, id)) => {
                out.push(id);
                start += wp;
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
    let (Some(vpath), Some(cpath)) = (
        std::env::var("WP_VOCAB").ok(),
        std::env::var("WP_CORPUS").ok(),
    ) else {
        println!(
            "wordpiece bench: set WP_VOCAB=<bert vocab.txt> WP_CORPUS=<text file> to run; skipping"
        );
        return;
    };
    let wp = load_vocab(&vpath);
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
    // Distinct token byte-lengths, descending — the "added-tokens" length set (V5).
    let mut lens_desc: Vec<u16> = wp.vocab.keys().map(|k| k.len() as u16).collect();
    lens_desc.sort_unstable();
    lens_desc.dedup();
    lens_desc.reverse();
    // First-byte-indexed length sets (V6 / shipped): lengths of tokens starting with byte b.
    let mut initial: Vec<Vec<u16>> = vec![Vec::new(); 256];
    let mut cont: Vec<Vec<u16>> = vec![Vec::new(); 256];
    for k in wp.vocab.keys() {
        let b = k.as_bytes();
        if b.is_empty() {
            continue;
        }
        initial[b[0] as usize].push(b.len() as u16);
        if b.len() > prefix.len() && b.starts_with(&prefix) {
            cont[b[prefix.len()] as usize].push(b.len() as u16);
        }
    }
    for v in initial.iter_mut().chain(cont.iter_mut()) {
        v.sort_unstable();
        v.dedup();
        v.reverse();
    }
    let avg_bucket = {
        let ne: Vec<usize> = initial
            .iter()
            .chain(cont.iter())
            .map(|v| v.len())
            .filter(|&n| n > 0)
            .collect();
        ne.iter().sum::<usize>() as f64 / ne.len().max(1) as f64
    };
    let words = load_corpus(&cpath, 200_000);
    let total: usize = words.iter().map(|w| w.len()).sum();

    // Byte-exactness of every variant against the shipped legacy path.
    let (mut base, mut v2, mut v3, mut v4, mut v5, mut v6, mut vp, mut tmp) = (
        Vec::new(),
        Vec::new(),
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
        v5.clear();
        greedy_lenset(w, &prefix, &lens_desc, unk, &store, &mut v5);
        v6.clear();
        greedy_lenset_fb(w, &prefix, &initial, &cont, unk, &store, &mut v6);
        tmp.clear();
        PipelineModel::tokenize_pipeline(&pw, w, &mut tmp).unwrap();
        vp.clear();
        vp.extend(tmp.iter().map(|t| t.id));

        if v2 != base || v3 != base || v4 != base || v5 != base || v6 != base || vp != base {
            ok = false;
            eprintln!("DIVERGE on {w:?}: base={base:?} v5={v5:?} v6={v6:?}");
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
    let (v5, s5) = time(&words, total, iters, |w, o| {
        greedy_lenset(w, &prefix, &lens_desc, unk, &store, o)
    });
    let (v6, s6) = time(&words, total, iters, |w, o| {
        greedy_lenset_fb(w, &prefix, &initial, &cont, unk, &store, o)
    });
    let (vp, s7) = time(&words, total, iters, |w, o| {
        let mut t = Vec::new();
        PipelineModel::tokenize_pipeline(&pw, w, &mut t).unwrap();
        o.extend(t.iter().map(|x| x.id));
    });
    black_box((s1, s2, s3, s4, s5, s6, s7));

    println!("\nWordPiece model lookup — {} vocab tokens, longest {max_token_len}B, {} words, {total} bytes", wp.vocab.len(), words.len());
    println!("  V1 legacy    (AHashMap, unbounded)      : {v1:>7.3} ns/byte  (1.00x)");
    println!("  V2 bounded   (AHashMap, bounded)        : {v2:>7.3} ns/byte  ({:.2}x)   isolates bounding", v1 / v2);
    println!("  V3 mphf      (BucketVocabStore, scalar) : {v3:>7.3} ns/byte  ({:.2}x)   isolates AHashMap->MPHF", v1 / v3);
    println!("  V4 streamed  (index_stream, prefetch)   : {v4:>7.3} ns/byte  ({:.2}x)   isolates streaming (x86 only)", v1 / v4);
    println!(
        "  V5 len-set   (probe {} distinct token lengths, MPHF) : {v5:>7.3} ns/byte  ({:.2}x)  <- shipped algorithm",
        lens_desc.len(),
        v1 / v5
    );
    println!("  V6 len-set/1B(probe ~{avg_bucket:.1} lengths for word[start]'s byte) : {v6:>7.3} ns/byte  ({:.2}x)  (slower on real data: per-byte indirection > probe savings)", v1 / v6);
    println!("  ** PipelineWordPiece end-to-end (V5 + PipelineToken output): {vp:>7.3} ns/byte  ({:.2}x) **", v1 / vp);
    println!(
        "  byte-exact across all variants: {}",
        if ok { "✓" } else { "✗ MISMATCH" }
    );
    assert!(ok, "a variant diverged from legacy");
}
