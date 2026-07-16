//! atomnorm vs the field: atomsplit::nfd (the POC it re-designs), unicode-normalization (Rust
//! reference), and xxUTF (C + SIMD, `--features xxutf`). Same corpora as atomsplit's bench.
//! run: cargo bench -p atomnorm --bench normalize [--features xxutf]
use std::borrow::Cow;
use std::hint::black_box;
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

#[cfg(feature = "xxutf")]
unsafe extern "C" {
    fn xxutf_normalize_utf8_nfd(input: *const u8, length: usize, out: *mut u8) -> usize;
    fn xxutf_normalize_utf8_nfkd(input: *const u8, length: usize, out: *mut u8) -> usize;
    fn xxutf_normalize_utf8_nfc(input: *const u8, length: usize, out: *mut u8) -> usize;
    fn xxutf_normalize_utf8_nfkc(input: *const u8, length: usize, out: *mut u8) -> usize;
}

type NormFn = fn(&str) -> Cow<'_, str>;
const FORMS: &[(&str, NormFn, NormFn)] = &[
    ("NFC", atomnorm::nfc as NormFn, atomnorm::scalar::nfc as NormFn),
    ("NFD", atomnorm::nfd, atomnorm::scalar::nfd),
    ("NFKC", atomnorm::nfkc, atomnorm::scalar::nfkc),
    ("NFKD", atomnorm::nfkd, atomnorm::scalar::nfkd),
];

fn unic(form: &str, s: &str) -> String {
    match form {
        "NFC" => s.nfc().collect(),
        "NFD" => s.nfd().collect(),
        "NFKC" => s.nfkc().collect(),
        _ => s.nfkd().collect(),
    }
}

#[cfg(feature = "xxutf")]
fn xxutf(form: &str, s: &str, out: &mut [u8]) -> usize {
    let (i, n, o) = (s.as_ptr(), s.len(), out.as_mut_ptr());
    // SAFETY: out is sized n*4+64 ≥ any expansion; input is valid UTF-8.
    unsafe {
        match form {
            "NFC" => xxutf_normalize_utf8_nfc(i, n, o),
            "NFD" => xxutf_normalize_utf8_nfd(i, n, o),
            "NFKC" => xxutf_normalize_utf8_nfkc(i, n, o),
            _ => xxutf_normalize_utf8_nfkd(i, n, o),
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
    ("French", "../atomsplit/benches/data/fr.txt"),
    ("Russian", "../atomsplit/benches/data/ru.txt"),
    ("Greek", "../atomsplit/benches/data/el.txt"),
    ("Hebrew", "../atomsplit/benches/data/he.txt"),
    ("Arabic", "../atomsplit/benches/data/ar.txt"),
    ("Hindi", "../atomsplit/benches/data/hi.txt"),
    ("Thai", "../atomsplit/benches/data/th.txt"),
    ("Chinese", "../atomsplit/benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("Korean", "../atomsplit/benches/data/ko.txt"),
];

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    println!(
        "\nns/byte, lower=better. simd = atomnorm (full); sclr = atomnorm scalar path; vsXX = simd vs xxUTF.\n\
         exact = BOTH atomnorm paths byte-identical to unicode-normalization.\n"
    );
    println!(
        "{:<5} {:<9} {:>7} {:>7} {:>7} {:>8} {:>8} | {:>7} {:>6} {:>6}",
        "form", "lang", "bytes", "simd", "sclr", "unic", "xxUTF", "vsUnic", "vsXX", "exact"
    );
    for (fname, an, asc) in FORMS {
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

            let reference = unic(fname, text);
            let exact = an(text).as_bytes() == reference.as_bytes()
                && asc(text).as_bytes() == reference.as_bytes();

            let t_norm = best(n, iters, || {
                black_box(an(text));
            });
            let t_sclr = best(n, iters, || {
                black_box(asc(text));
            });
            let t_unic = best(n, iters, || {
                black_box(unic(fname, text));
            });
            #[cfg(feature = "xxutf")]
            let t_xx: Option<f64> = {
                let mut out = vec![0u8; n * 4 + 64];
                Some(best(n, iters, || {
                    black_box(xxutf(fname, text, &mut out));
                }))
            };
            #[cfg(not(feature = "xxutf"))]
            let t_xx: Option<f64> = None;

            let xxc = t_xx.map_or("       —".into(), |t| format!("{t:>8.3}"));
            let vsxx = t_xx.map_or("     —".into(), |t| format!("{:>5.1}x", t / t_norm));
            println!(
                "{fname:<5} {label:<9} {n:>7} {t_norm:>7.3} {t_sclr:>7.3} {t_unic:>8.3} {xxc} | {:>6.1}x {vsxx} {:>6}",
                t_unic / t_norm,
                if exact { "✓" } else { "✗" }
            );
        }
    }
}
