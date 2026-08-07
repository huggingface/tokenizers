//! Throughput per grammar. `cargo run --release -p bitsplit --example grammars`
use bitsplit::Span;
use bitsplit::classify::classify;
use std::time::Instant;

type Split = fn(&[u8], &[u8], &mut [u64], &mut [u64], &mut [Span]) -> usize;

fn ds(t: &[u8], g: &[u8], s: &mut [u64], _f: &mut [u64], o: &mut [Span]) -> usize {
    bitsplit::bitsplit_deepseek(t, g, s, o)
}

fn main() {
    let corpora: Vec<(&str, String)> = vec![
        ("english", "The quick brown fox jumps over the lazy dog. Don't stop, it's 42 times! ".repeat(20000)),
        ("code", "fn main() { let x: Vec<u32> = (0..10).map(|i| i * 2).collect(); }\n".repeat(20000)),
        ("chinese", "中文分词测试，这是一个很长的句子。".repeat(40000)),
        ("mixed", "Hello 世界 café Привет 123 !!! \n\n  tabs\there\n".repeat(20000)),
    ];
    let grammars: Vec<(&str, Split)> = vec![
        ("gpt2", bitsplit::bitsplit_byte_level),
        ("cl100k", bitsplit::bitsplit_cl100k),
        ("qwen", bitsplit::bitsplit_qwen),
        ("o200k", bitsplit::bitsplit_o200k),
        ("tekken", bitsplit::bitsplit_tekken),
        ("kimi", bitsplit::bitsplit_kimi),
        ("deepseek", ds),
    ];
    print!("{:<10}", "MB/s");
    for (name, _) in &corpora { print!("{name:>10}"); }
    println!();
    for (gname, f) in &grammars {
        print!("{gname:<10}");
        for (_, text) in &corpora {
            let b = text.as_bytes();
            let mut tags = vec![0u8; b.len()];
            let words = b.len().div_ceil(64) + 1;
            let (mut st, mut fl) = (vec![0u64; words], vec![0u64; words]);
            let mut out = vec![Span::default(); b.len() + 1];
            classify(b, &mut tags); // measured separately; this is the model-facing split only
            let mut k = 0;
            let t = Instant::now();
            for _ in 0..3 { k = f(b, &tags, &mut st, &mut fl, &mut out); }
            let secs = t.elapsed().as_secs_f64() / 3.0;
            std::hint::black_box(k);
            print!("{:>10.0}", b.len() as f64 / (1 << 20) as f64 / secs);
        }
        println!();
    }
}
