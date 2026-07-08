//! cl100k_base pretokenization, PER SCRIPT (no aggregation) across UTF-8 byte-widths.
//! For each script: classify (SIMD vs scalar) and fsm (scalar vs SIMD run-end), in MB/s, plus a
//! byte-exactness check of the full pipeline vs the onig cl100k regex. ASCII shows the fast-path
//! ceiling; multibyte scripts show the 2/3/4-byte lane cost — the aggregate hides both.
//!
//! Run: cargo bench --bench cl100k
use fast_split::classify::{Atoms, classify, classify_scalar};
#[cfg(target_arch = "aarch64")]
use fast_split::fsm::fsm_cl100k_simd;
use fast_split::fsm::{Span, fsm_cl100k};
use onig::Regex;
use std::hint::black_box;
use std::time::Instant;

const CL100K: &str = concat!(
    r"'(?i:[sdmt]|ll|ve|re)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]|\s+(?!\S)|\s+",
);

const SCRIPTS: &[(&str, &str)] = &[
    ("English  1B", "The quick brown fox jumps over 13 lazy dogs; don't you think? "),
    ("code     1B", "fn main() { let x = vec![1,2,3]; println!(\"{}\", x.len()); }\n"),
    ("French   2B", "Portez ce vieux whisky au juge blond qui fume — 42% déjà. "),
    ("Russian  2B", "Съешь же ещё этих мягких французских булочек да выпей чаю. "),
    ("Greek    2B", "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία. "),
    ("Hebrew   2B", "דג סקרן שט בים מאוכזב ולפתע מצא חברה. "),
    ("Arabic   2B", "نص حكيم له سر قاطع وذو شأن عظيم. "),
    ("Hindi    3B", "ऋषियों को सताने वाले दुष्ट राक्षसों के राजा का सर्वनाश। "),
    ("Thai     3B", "เป็นมนุษย์สุดประเสริฐเลิศคุณค่า "),
    ("Chinese  3B", "視野無限廣，窗外有藍天。快速跳躍123。 "),
    ("Japanese 3B", "いろはにほへと ちりぬるを カタカナ 漢字。 "),
    ("Korean   3B", "다람쥐 헌 쳇바퀴에 타고파 42번. "),
    ("Emoji    4B", "🦊🚀🔥😀🎉 test 007 "),
];

fn ns_per_byte<F: FnMut() -> usize>(len: usize, iters: u32, mut f: F) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let t = Instant::now();
    let mut acc = 0usize;
    for _ in 0..iters {
        acc = acc.wrapping_add(f());
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / (iters as usize * len) as f64
}

fn main() {
    let re = Regex::new(CL100K).expect("cl100k regex");
    let iters = 400;
    let mbps = |ns: f64| 1000.0 / ns;

    // MB/s columns; classify (SIMD/scalar), fsm (scalar/SIMD run-end), pipeline SIMD vs onig.
    #[cfg(target_arch = "aarch64")]
    println!(
        "{:<12} {:>7} {:>4}  {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>7}",
        "script", "bytes", "b/ch", "clsSIMD", "clsScal", "fsmScal", "fsmSIMD", "onig", "vsOnig"
    );

    for (name, unit) in SCRIPTS {
        let corpus = unit.repeat(220);
        let text = corpus.as_bytes();
        let n = text.len();
        let bpc = n as f64 / corpus.chars().count() as f64;

        // correctness: full pipeline == onig
        let onig: Vec<Span> = re.find_iter(&corpus).map(|(s, e)| (s as u32, e as u32)).collect();
        let mut tags = vec![0u8; n];
        classify::<Atoms>(text, &mut tags);
        let mut sc = Vec::new();
        fsm_cl100k(text, &tags, &mut sc);
        let ok = if sc == onig { "✓" } else { "✗" };

        // per-pass timing
        let mut tsc = vec![0u8; n];
        let mut buf = Vec::with_capacity(sc.len());
        let cls_simd = ns_per_byte(n, iters, || {
            classify::<Atoms>(text, &mut tags);
            tags[n / 2] as usize
        });
        let cls_scal = ns_per_byte(n, iters, || {
            classify_scalar::<Atoms>(text, &mut tsc);
            tsc[n / 2] as usize
        });
        classify::<Atoms>(text, &mut tags);
        let fsm_scal = ns_per_byte(n, iters, || {
            buf.clear();
            fsm_cl100k(text, &tags, &mut buf);
            buf.len()
        });
        let onig_ns = ns_per_byte(n, iters, || re.find_iter(&corpus).count());
        #[cfg(target_arch = "aarch64")]
        {
            let fsm_simd = ns_per_byte(n, iters, || {
                buf.clear();
                fsm_cl100k_simd(text, &tags, &mut buf);
                buf.len()
            });
            let pipe = cls_simd + fsm_simd; // SIMD classify + SIMD fsm
            println!(
                "{name:<12} {n:>7} {bpc:>4.1}  {:>8.0} {:>8.0} | {:>8.0} {:>8.0} | {:>8.1} {:>6.1}x {ok}",
                mbps(cls_simd),
                mbps(cls_scal),
                mbps(fsm_scal),
                mbps(fsm_simd),
                mbps(onig_ns),
                onig_ns / pipe
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = ok;
            println!(
                "{name:<12} {n:>7} clsSIMD {:>6.0} clsScal {:>6.0} fsmScal {:>6.0} onig {:>5.1} MB/s",
                mbps(cls_simd),
                mbps(cls_scal),
                mbps(fsm_scal),
                mbps(onig_ns)
            );
        }
    }
    println!("\n(MB/s; higher is better. clsSIMD ceiling is the ASCII fast path; b/ch = bytes per char.)");
}
