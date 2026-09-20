//! Differential test for the class-run family: the bitstream `class_runs_into` against the scalar
//! FSM it replaced. Truncating the corpus at every char boundary walks every construct through
//! every block phase, which is the only thing that exercises the cross-block carries.

use bitsplit::Span;
use bitsplit::classify::{CONT, char_len, classify, in_mask, mask};

/// The scalar class-run FSM, verbatim from `classes.rs` before the bitstream port.
fn reference(text: &[u8], tags: &[u8], dropm: u16, isolatem: u16, keepa: u16) -> Vec<Span> {
    let n = text.len();
    let other = !(dropm | isolatem | keepa);
    let run_end = |mut i: usize, m: u16| {
        let m = m | 1u16 << CONT;
        while i < n && in_mask(tags[i], m) {
            i += 1;
        }
        i
    };
    let (mut i, mut out) = (0usize, Vec::new());
    while i < n {
        let t = tags[i];
        if t == CONT {
            i += 1;
        } else if in_mask(t, dropm) {
            i = run_end(i, dropm);
        } else if in_mask(t, isolatem) {
            let s = i;
            i += char_len(text[i]);
            out.push(Span::new(s as u32, i as u32));
        } else {
            let s = i;
            i = if in_mask(t, keepa) {
                run_end(i, keepa)
            } else {
                run_end(i, other)
            };
            out.push(Span::new(s as u32, i as u32));
        }
    }
    out
}

fn check<const D: u16, const I: u16, const K: u16>(name: &str, text: &[u8]) {
    let mut tags = vec![0u8; text.len()];
    classify(text, &mut tags);
    let want = reference(text, &tags, D, I, K);
    let words = text.len().div_ceil(64) + 1;
    let (mut st, mut fk) = (vec![0u64; words], vec![0u64; words]);
    let mut got = vec![Span::default(); text.len() + 1];
    let k = bitsplit::classes::class_runs_into::<D, I, K>(text, &tags, &mut st, &mut fk, &mut got);
    assert_eq!(
        &got[..k],
        &want[..],
        "{name} D={D:#06x} I={I:#06x} K={K:#06x} len={}",
        text.len()
    );
}

/// Every live instantiation across `tk-encode`'s pre-tokenizers.
fn all_combos(name: &str, t: &[u8]) {
    check::<{ mask::WS }, 0, 0>(name, t); // WhitespaceSplit
    check::<0, { mask::PUNCT }, 0>(name, t); // Punctuation, Isolated
    check::<{ mask::PUNCT }, 0, 0>(name, t); // Punctuation, Removed
    check::<0, 0, { mask::NUMERIC }>(name, t); // Digits, runs
    check::<0, { mask::NUMERIC }, 0>(name, t); // Digits, individual
    check::<{ mask::WS }, 0, { mask::WORD }>(name, t); // Whitespace `\w+|[^\w\s]+`
    check::<{ mask::WS }, { mask::PUNCT }, 0>(name, t); // Bert
}

/// Mixed enough that every branch is live, and long enough to span several blocks.
const MIXED: &str = "Hello, world! 42 items — naïve café 3.14\tX\r\n\n  ¡Hola!  \
     日本語テスト、句読点。 Ⅻ ½ ٣٤٥ देवनागरी ｆｕｌｌ width 007 a_b-c'd \
     tabs\t\tand   spaces, ends with digits 12345";

#[test]
fn block_phase_sweep() {
    let s = MIXED;
    for (i, _) in s.char_indices().chain(std::iter::once((s.len(), ' '))) {
        all_combos("MIXED", &s.as_bytes()[..i]);
    }
}

#[test]
fn corpora() {
    let dir = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tokbench/data/fixtures"
    ));
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("skip: no fixtures at {}", dir.display());
        return;
    };
    let mut files: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "fixture dir exists but has no .txt");
    for p in &files {
        let Ok(s) = std::fs::read_to_string(p) else {
            continue;
        };
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        // a few lengths, each cut to a char boundary
        for frac in [1usize, 3, 17, 4096] {
            let want = (s.len() / frac).min(1 << 20);
            let end = (0..=want)
                .rev()
                .find(|&i| s.is_char_boundary(i))
                .unwrap_or(0);
            all_combos(&name, &s.as_bytes()[..end]);
        }
    }
    eprintln!("{} corpora x 4 lengths x 7 combos ok", files.len());
}
