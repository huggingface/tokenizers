//! UnicodeScripts: our atom-engine pre-tokenizer (`classify::<Scripts>` → `fsm_script_run`) vs a faithful
//! replica of the PREVIOUS pretokenizer's algorithm (tk-encode's `pipeline` impl: a BMP script cache +
//! script-change scan). Both use the SAME `unicode-script` data, so the parity column isolates the ONE
//! semantic difference: the old impl treats `Common`/`Inherited` (digits, punctuation, `\n`, `。`) as
//! REAL splitting scripts, while our `bitmap_gen` table folds them into the transparent set (id 0).
//! Only `' '` and unassigned are transparent in the old impl.
//!
//! Run: cargo bench --features unicode-scripts --bench unicode_scripts
use fast_split::fsm::{Span, UnicodeScripts};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;
use unicode_script::{Script, UnicodeScript};

const CORPORA: &[(&str, &str)] = &[
    ("English", "../data/big.txt"),
    ("French", "benches/data/fr.txt"),
    ("Russian", "benches/data/ru.txt"),
    ("Greek", "benches/data/el.txt"),
    ("Arabic", "benches/data/ar.txt"),
    ("Hebrew", "benches/data/he.txt"),
    ("Hindi", "benches/data/hi.txt"),
    ("Thai", "benches/data/th.txt"),
    ("Chinese", "benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("Korean", "benches/data/ko.txt"),
];

/// Old `fixed_script` semantics as a compact per-codepoint id (0 = transparent = old `Script::Any`):
///   ` ` → 0 · unassigned (`Unknown`) → 0 · `0x30FC` → Han · Hira/Kata → Han · everything else (incl.
///   `Common`/`Inherited`) → its own real id. Precomputed once, like tk-encode's `BMP_SCRIPT` cache.
fn old_id_table() -> Vec<u16> {
    let mut ids = vec![0u16; 0x11_0000];
    let mut map: HashMap<Script, u16> = HashMap::new();
    let mut next: u16 = 0;
    for cp in 0u32..=0x10FFFF {
        let Some(ch) = char::from_u32(cp) else { continue };
        if ch == ' ' {
            continue; // → 0 (Any)
        }
        let s = if cp == 0x30FC {
            Script::Han
        } else {
            match ch.script() {
                Script::Hiragana | Script::Katakana => Script::Han,
                Script::Unknown => continue, // → 0 (Any)
                other => other,              // Common / Inherited / real scripts are all REAL here
            }
        };
        ids[cp as usize] = *map.entry(s).or_insert_with(|| {
            next += 1;
            next
        });
    }
    ids
}

/// The previous pretokenizer's scan: split on real-script change; id 0 (Any) is transparent and sticks
/// to the surrounding run. Mirrors `tk-encode`'s `pipeline::PreTokenizer for UnicodeScripts`.
fn old_scripts(text: &str, ids: &[u16], out: &mut Vec<Span>) {
    out.clear();
    let mut start: Option<u32> = None;
    let mut last: u16 = 0; // 0 = no real script seen yet
    for (i, ch) in text.char_indices() {
        let id = ids[ch as usize];
        if id == 0 {
            continue;
        }
        if last != id {
            if let Some(s) = start {
                out.push((s, i as u32));
            }
            start = Some(i as u32);
        }
        last = id;
    }
    if let Some(s) = start {
        out.push((s, text.len() as u32));
    }
}

fn ns_per_byte<F: FnMut() -> usize>(len: usize, iters: u32, mut f: F) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        let mut acc = 0usize;
        for _ in 0..iters {
            acc = acc.wrapping_add(f());
        }
        black_box(acc);
        best = best.min(t.elapsed().as_nanos() as f64 / (iters as usize * len) as f64);
    }
    best
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let ids = old_id_table(); // one-time (mirrors tk-encode's BMP cache); not timed
    println!(
        "{:<10} {:>7} {:>6} {:>6}  {:>7} {:>7} | {:>7} {:>7}",
        "lang", "bytes", "b/tok", "old/tk", "ours", "old", "speedup", "parity"
    );

    for (label, rel) in CORPORA {
        let raw = match std::fs::read_to_string(format!("{manifest}/{rel}")) {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                println!("{label:<10}  (skipped — {rel} missing)");
                continue;
            }
        };
        let mut c = raw.len().min(180_000);
        while c > 0 && !raw.is_char_boundary(c) {
            c -= 1;
        }
        let corpus = &raw[..c];
        let text = corpus.as_bytes();
        let n = text.len();
        let iters = (4_000_000 / n).clamp(3, 150) as u32;

        // outputs + parity (ours folds Common/Inherited → transparent; old keeps them as real scripts)
        let mut tags = vec![0u8; n];
        let (mut ours, mut old) = (Vec::new(), Vec::new());
        UnicodeScripts.pre_tokenize(text, &mut tags, &mut ours);
        old_scripts(corpus, &ids, &mut old);
        let matched = ours.iter().filter(|s| old.binary_search(s).is_ok()).count();
        let parity = 100.0 * matched as f64 / ours.len().max(1) as f64;
        let btok = n as f64 / ours.len().max(1) as f64;
        let btok_old = n as f64 / old.len().max(1) as f64;

        let t_ours = ns_per_byte(n, iters, || {
            ours.clear();
            UnicodeScripts.pre_tokenize(text, &mut tags, &mut ours);
            ours.len()
        });
        let t_old = ns_per_byte(n, iters, || {
            old_scripts(corpus, &ids, &mut old);
            old.len()
        });

        println!(
            "{label:<10} {n:>7} {btok:>6.1} {btok_old:>6.1}  {t_ours:>7.3} {t_old:>7.3} | {:>6.2}x {parity:>6.1}%",
            t_old / t_ours
        );
    }
    println!(
        "\n(ns/byte, lower better. ours = classify::<Scripts> + fsm_script_run; old = tk-encode script-change\n \
         scan. b/tok = bytes per token. parity = % of our spans the old impl also produces — <100% is the\n \
         Common/Inherited transparency difference, not a perf artifact.)"
    );
}
