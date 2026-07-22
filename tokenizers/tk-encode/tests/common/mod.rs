//! Shared setup for the pipeline oracle tests. Encode and decode parity are
//! judged over the same inputs: seeded-random windows of the real fixture corpora
//! (`data/fixtures/{lang,modalities}` — 14 languages + code/math/agentic). Random
//! for coverage across varied text, seeded so any failure reproduces exactly.
//! Both oracles generate one `#[test]` per fixture via [`for_each_fixture`], so a
//! red run names the exact model and language/modality that broke.

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

/// Deterministic per-fixture seed — FNV-1a of the stem, so each fixture samples a
/// different slice of its corpus while any failure still reproduces exactly.
pub fn stem_seed(stem: &str) -> u64 {
    stem.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
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

/// One `#[test]` per fixture: expands `$cb($($extra,)* group, stem)` for every
/// fixture in `data/fixtures/{lang,modalities}`. The list mirrors the Makefile's
/// `FIXTURE_LANGS`/`FIXTURE_MODALITIES`; absent files skip inside the callback.
/// `ident = "stem"` overrides the file stem when it isn't a valid identifier.
macro_rules! for_each_fixture {
    ($cb:path $(, $extra:expr)* $(,)?) => {
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", amh_Ethi);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", arb_Arab);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", ben_Beng);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", cmn_Hani);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", ell_Grek);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", eng_Latn);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", heb_Hebr);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", hin_Deva);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", jpn_Jpan);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", kat_Geor);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", kor_Hang);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", rus_Cyrl);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", tam_Taml);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"lang", tha_Thai);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"modalities", added_normalized_dense);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"modalities", added_normalized_sparse);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"modalities", added_special_dense);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"modalities", added_special_sparse);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"modalities", agentic_swe);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"modalities", agentic_traces = "agentic-traces");
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"modalities", code_mixed);
        crate::common::for_each_fixture!(@one $cb, ($($extra),*),"modalities", math_latex);
    };
    (@one $cb:path, ($($extra:expr),*), $group:literal, $name:ident) => {
        #[test]
        fn $name() {
            $cb($($extra,)* $group, stringify!($name));
        }
    };
    (@one $cb:path, ($($extra:expr),*), $group:literal, $name:ident = $stem:literal) => {
        #[test]
        fn $name() {
            $cb($($extra,)* $group, $stem);
        }
    };
}
pub(crate) use for_each_fixture;
