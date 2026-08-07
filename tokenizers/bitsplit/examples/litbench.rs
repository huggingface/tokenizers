//! Literal bitstream vs memmem on the metaspace needle. `cargo run --release -p bitsplit --example litbench`
use std::time::Instant;
fn main() {
    // metaspace-shaped text: words separated by U+2581, ~6 chars/word
    let mut text = String::new();
    while text.len() < 8 << 20 {
        text.push_str("\u{2581}the\u{2581}quick\u{2581}brown\u{2581}fox\u{2581}jumps\u{2581}over123");
    }
    let b = text.as_bytes();
    let needle = "\u{2581}".as_bytes();
    let lit = bitsplit::literal::Literal::new(needle).unwrap();

    let mut n1 = 0usize;
    let t = Instant::now();
    for _ in 0..5 { n1 = lit.matches(b).count(); }
    let bits = t.elapsed().as_secs_f64() / 5.0;

    let f = memchr::memmem::Finder::new(needle);
    let mut n2 = 0usize;
    let t = Instant::now();
    for _ in 0..5 { n2 = f.find_iter(b).count(); }
    let mm = t.elapsed().as_secs_f64() / 5.0;

    assert_eq!(n1, n2, "different match counts");
    let mb = b.len() as f64 / (1 << 20) as f64;
    println!("{:.1} MB, {n1} matches", mb);
    println!("  bitstream {:>7.1} MB/s  ({:.1} ms)", mb / bits, bits * 1e3);
    println!("  memmem    {:>7.1} MB/s  ({:.1} ms)", mb / mm, mm * 1e3);
    println!("  ratio     {:>7.2}x", mm / bits);
}
