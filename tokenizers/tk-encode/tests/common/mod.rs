//! Shared setup for the pipeline oracle tests. Encode and decode parity are
//! judged over the same inputs: seeded-random windows of the real fixture corpora
//! (`data/fixtures/{lang,modalities}` — 14 languages + code/math/agentic). Random
//! for coverage across varied text, seeded so any failure reproduces exactly.

use std::path::{Path, PathBuf};

/// Test-data root, shared with the benchmark harness (populated by `make fixtures`).
pub const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

/// Window sizes sampled per fixture: one per-input-overhead-sized, one amortized.
pub const WINDOWS: &[usize] = &[1024, 8 * 1024];

/// splitmix64 — a tiny deterministic PRNG so the "random" offsets stay reproducible.
pub fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Existing `.txt` fixtures under `data/fixtures/{lang,modalities}`, sorted. Empty
/// when fixtures haven't been fetched — callers skip rather than fail.
pub fn fixture_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for group in ["lang", "modalities"] {
        let dir = Path::new(DATA).join("fixtures").join(group);
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut paths: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "txt"))
                .collect();
            paths.sort();
            out.extend(paths);
        }
    }
    out
}

/// A `window`-byte slice of `text` from a seeded offset, snapped to char boundaries.
pub fn random_chunk(text: &str, window: usize, seed: u64) -> &str {
    let len = text.len();
    if len <= window {
        return text;
    }
    let mut start = splitmix64(seed) as usize % (len - window);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    let mut end = (start + window).min(len);
    while end < len && !text.is_char_boundary(end) {
        end += 1;
    }
    &text[start..end]
}
