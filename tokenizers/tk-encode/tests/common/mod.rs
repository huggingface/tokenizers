//! Shared setup for the pipeline oracle tests. Encode and decode parity are
//! judged over the same inputs: fixed-size windows of the real fixture corpora
//! (`data/fixtures/{lang,modalities}` — 14 languages + code/math/agentic).

/// Test-data root, shared with the benchmark harness (populated by `make fixtures`).
pub const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

/// Window sizes taken per fixture, consecutively from the start of the file:
/// one per-input-overhead-sized, one amortized.
pub const WINDOWS: &[usize] = &[1024, 8 * 1024];

/// Every fixture `make fixtures` fetches, as `(group, file stem)` — mirrors the
/// Makefile's `FIXTURE_LANGS`/`FIXTURE_MODALITIES`. Absent files skip at run time.
pub const FIXTURES: &[(&str, &str)] = &[
    ("lang", "amh_Ethi"),
    ("lang", "arb_Arab"),
    ("lang", "ben_Beng"),
    ("lang", "cmn_Hani"),
    ("lang", "ell_Grek"),
    ("lang", "eng_Latn"),
    ("lang", "heb_Hebr"),
    ("lang", "hin_Deva"),
    ("lang", "jpn_Jpan"),
    ("lang", "kat_Geor"),
    ("lang", "kor_Hang"),
    ("lang", "rus_Cyrl"),
    ("lang", "tam_Taml"),
    ("lang", "tha_Thai"),
    ("modalities", "added_normalized_dense"),
    ("modalities", "added_normalized_sparse"),
    ("modalities", "added_special_dense"),
    ("modalities", "added_special_sparse"),
    ("modalities", "agentic_swe"),
    ("modalities", "agentic-traces"),
    ("modalities", "code_mixed"),
    ("modalities", "math_latex"),
];

/// `len` bytes of `text` from `start`, both ends snapped to char boundaries.
pub fn window(text: &str, start: usize, len: usize) -> &str {
    if start >= text.len() {
        return "";
    }
    let mut s = start;
    while !text.is_char_boundary(s) {
        s += 1;
    }
    let mut e = (s + len).min(text.len());
    while e < text.len() && !text.is_char_boundary(e) {
        e += 1;
    }
    &text[s..e]
}
