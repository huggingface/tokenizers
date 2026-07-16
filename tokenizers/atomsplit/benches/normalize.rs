//! Unicode normalization throughput — atomsplit's pure-Rust `nfd` (NFD / NFKD / NFC / NFKC) against the
//! two obvious references:
//!   * `unicode-normalization` (the de-facto Rust crate; also the differential-test oracle for `src/nfd.rs`)
//!   * `xxUTF` (dzfrias/xxUTF — a C + SIMD normalizer, "fastest open source"), behind `--features xxutf`
//!     (a build script downloads + compiles the pinned MIT amalgamation; default builds stay C-free).
//!
//! Same corpora as the `regex` bench (`../data` + `benches/data`, via `benches/data/fetch.py`; missing
//! files are skipped). min-of-7 ns/byte per form × corpus, plus a byte-exactness gate vs
//! `unicode-normalization`. atomsplit borrows already-normalized input (the common case), so on NFC/NFKC
//! over real (already-NFC) text it does almost no work — that borrow is the whole point, and it shows.
//!
//! run: cargo bench --bench normalize                 # atomsplit vs unicode-normalization
//!      cargo bench --bench normalize --features xxutf # also vs xxUTF (needs a C compiler + network once)
use std::borrow::Cow;
use std::hint::black_box;
use std::panic;
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

// xxUTF C API (utf-8): `size_t xxutf_normalize_utf8_FORM(const char *in, size_t len, char *out)` → out len.
#[cfg(feature = "xxutf")]
unsafe extern "C" {
    fn xxutf_normalize_utf8_nfd(input: *const u8, length: usize, out: *mut u8) -> usize;
    fn xxutf_normalize_utf8_nfkd(input: *const u8, length: usize, out: *mut u8) -> usize;
    fn xxutf_normalize_utf8_nfc(input: *const u8, length: usize, out: *mut u8) -> usize;
    fn xxutf_normalize_utf8_nfkc(input: *const u8, length: usize, out: *mut u8) -> usize;
}

#[derive(Clone, Copy)]
enum Form {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
}
use Form::*;
const FORMS: &[(&str, Form)] = &[("NFC", Nfc), ("NFD", Nfd), ("NFKC", Nfkc), ("NFKD", Nfkd)];

fn atom(f: Form, s: &str) -> Cow<'_, str> {
    match f {
        Nfc => atomsplit::nfd::nfc(s),
        Nfd => atomsplit::nfd::nfd(s),
        Nfkc => atomsplit::nfd::nfkc(s),
        Nfkd => atomsplit::nfd::nfkd(s),
    }
}

fn unic(f: Form, s: &str) -> String {
    match f {
        Nfc => s.nfc().collect(),
        Nfd => s.nfd().collect(),
        Nfkc => s.nfkc().collect(),
        Nfkd => s.nfkd().collect(),
    }
}

#[cfg(feature = "xxutf")]
fn xxutf(f: Form, s: &str, out: &mut [u8]) -> usize {
    let (i, n, o) = (s.as_ptr(), s.len(), out.as_mut_ptr());
    // SAFETY: `out` is sized `s.len()*4 + 64` ≥ any NFKD/NFKC expansion of `s`; input is valid UTF-8.
    unsafe {
        match f {
            Nfc => xxutf_normalize_utf8_nfc(i, n, o),
            Nfd => xxutf_normalize_utf8_nfd(i, n, o),
            Nfkc => xxutf_normalize_utf8_nfkc(i, n, o),
            Nfkd => xxutf_normalize_utf8_nfkd(i, n, o),
        }
    }
}

fn best(len: usize, iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        b = b.min(t.elapsed().as_nanos() as f64 / (iters as usize * len) as f64);
    }
    b
}

const CORPORA: &[(&str, &str)] = &[
    ("English", "../data/big.txt"),
    ("French", "benches/data/fr.txt"),
    ("Russian", "benches/data/ru.txt"),
    ("Greek", "benches/data/el.txt"),
    ("Hebrew", "benches/data/he.txt"),
    ("Arabic", "benches/data/ar.txt"),
    ("Hindi", "benches/data/hi.txt"),
    ("Thai", "benches/data/th.txt"),
    ("Chinese", "benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("Korean", "benches/data/ko.txt"),
];

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let xxutf_on = cfg!(feature = "xxutf");
    panic::set_hook(Box::new(|_| {})); // atomsplit's WIP nfd panics on some inputs; keep the table clean
    println!(
        "\nns/byte, lower=better. vsUnic / vsXX = how many× faster atomsplit is than unicode-normalization \
         / xxUTF. exact = atomsplit output byte-identical to unicode-normalization{}.\n(atomsplit borrows \
         already-normalized input — near-zero on NFC/NFKC over real text; NFD/NFKD force real decomposition.)\n",
        if xxutf_on { "; xok = xxUTF matches it (≈ tolerated — xxUTF tracks a newer Unicode)" } else { " (xxUTF off — pass --features xxutf)" }
    );
    println!(
        "{:<5} {:<9} {:>7} {:>8} {:>8} {:>8} | {:>7} {:>6} {:>6}{}",
        "form", "lang", "bytes", "atom", "unic", "xxUTF", "vsUnic", "vsXX", "exact", if xxutf_on { " xok" } else { "" }
    );
    for (fname, form) in FORMS {
        for (label, rel) in CORPORA {
            let Ok(s) = std::fs::read_to_string(format!("{manifest}/{rel}")) else { continue };
            if s.trim().is_empty() {
                continue;
            }
            let mut c = s.len().min(180_000);
            while c > 0 && !s.is_char_boundary(c) {
                c -= 1;
            }
            let text = &s[..c];
            let n = text.len();
            let iters = (4_000_000 / n).clamp(3, 150) as u32;

            let reference = unic(*form, text);
            // atomsplit's nfd is WIP and can panic on some inputs; guard it so one crash doesn't abort the
            // whole bench — a panicking cell is flagged PANIC and its timing skipped.
            let (ta, exact): (Option<f64>, bool) =
                match panic::catch_unwind(|| atom(*form, text).into_owned()) {
                    Ok(a) => (
                        Some(best(n, iters, || {
                            black_box(atom(*form, text));
                        })),
                        a.as_bytes() == reference.as_bytes(),
                    ),
                    Err(_) => (None, false),
                };
            let tu = best(n, iters, || {
                black_box(unic(*form, text));
            });

            #[cfg(feature = "xxutf")]
            let (tx, xok): (Option<f64>, bool) = {
                let mut out = vec![0u8; n * 4 + 64];
                let xn = xxutf(*form, text, &mut out);
                let xok = out[..xn] == *reference.as_bytes();
                let t = best(n, iters, || {
                    black_box(xxutf(*form, text, &mut out));
                });
                (Some(t), xok)
            };
            #[cfg(not(feature = "xxutf"))]
            let (tx, xok): (Option<f64>, bool) = (None, false);

            let atomc = ta.map_or_else(|| "   PANIC".to_string(), |t| format!("{t:>8.3}"));
            let xcell = tx.map_or_else(|| "       —".to_string(), |t| format!("{t:>8.3}"));
            let vsunic = ta.map_or_else(|| "     —".to_string(), |t| format!("{:>5.1}x", tu / t));
            let vsxx = match (ta, tx) {
                (Some(a), Some(x)) => format!("{:>5.1}x", x / a),
                _ => "     —".to_string(),
            };
            let exactc = if ta.is_none() {
                "PANIC"
            } else if exact {
                "✓"
            } else {
                "✗"
            };
            let xokc = if xxutf_on {
                if xok { "   ✓" } else { "   ≈" }
            } else {
                ""
            };
            println!(
                "{fname:<5} {label:<9} {n:>7} {atomc} {tu:>8.3} {xcell} | {vsunic} {vsxx} {exactc:>6}{xokc}"
            );
        }
    }
}
