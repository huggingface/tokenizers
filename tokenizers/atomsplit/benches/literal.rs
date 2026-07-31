//! `Literal::matches` (one search per match) against `Literal::matches_into` (one scan per
//! text), across the delimiter densities pre-tokenizers see: a space about every six bytes of
//! English, a `▁` per word after a SentencePiece replace, and a pattern the text does not
//! contain at all (where `memmem`'s skip-ahead is at its best and the scan must keep up).
//! Run: cargo bench --bench literal
use atomsplit::literal::Literal;
use std::hint::black_box;
use std::time::Instant;

fn best_ns_per_byte(text_len: usize, mut pass: impl FnMut() -> usize) -> f64 {
    let iters = (16_000_000 / text_len.max(1)).clamp(4, 400) as u32;
    for _ in 0..3 {
        black_box(pass());
    }
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        for _ in 0..iters {
            black_box(pass());
        }
        best = best.min(t.elapsed().as_nanos() as f64 / (iters as usize * text_len) as f64);
    }
    best
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let english =
        std::fs::read_to_string(format!("{manifest}/../data/big.txt")).unwrap_or_default();
    let english: String = english.chars().take(2_000_000).collect();
    if english.is_empty() {
        println!("empty corpus — data/big.txt missing? (make test downloads it)");
        return;
    }
    let metaspaced = english.replace(' ', "\u{2581}");

    println!(
        "{:<22} {:>9} {:>12} {:>12} {:>9} {:>12}",
        "", "matches", "iterator", "batch", "speedup", "count-only"
    );
    for (label, text, pattern) in [
        ("space in English", english.as_bytes(), " "),
        ("▁ in metaspaced", metaspaced.as_bytes(), "\u{2581}"),
        ("▁ absent", english.as_bytes(), "\u{2581}"),
    ] {
        let literal = Literal::new(pattern.as_bytes()).unwrap();
        let mut offsets: Vec<usize> = Vec::with_capacity(text.len());
        let mut buffer = vec![0u32; text.len() + 4];

        let count = literal.matches(text).count();
        let iterator = best_ns_per_byte(text.len(), || {
            offsets.clear();
            offsets.extend(literal.matches(text));
            offsets.len()
        });
        let batch = best_ns_per_byte(text.len(), || literal.matches_into(text, &mut buffer));
        let counting = best_ns_per_byte(text.len(), || literal.count_matches(text));

        println!(
            "{label:<22} {count:>9} {:>7.2} GB/s {:>7.2} GB/s {:>8.2}x {:>7.2} GB/s",
            1.0 / iterator,
            1.0 / batch,
            iterator / batch,
            1.0 / counting
        );
    }
}
