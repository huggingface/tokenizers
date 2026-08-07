#![cfg(not(target_arch = "wasm32"))]
use bitsplit::Span;
use bitsplit::classify::classify;
use bitsplit::regexes::O200K;
use onig::Regex;
fn spans(t: &str) -> Vec<Span> {
    if t.is_empty() { return vec![]; }
    let b = t.as_bytes();
    let mut tags = vec![0u8; b.len()];
    classify(b, &mut tags);
    let nblk = b.len().div_ceil(64);
    let (mut st, mut fl) = (vec![0u64; nblk], vec![0u64; nblk]);
    let mut out = vec![Span::default(); b.len() + 1];
    let n = bitsplit::bitsplit_o200k(b, &tags, &mut st, &mut fl, &mut out);
    out.truncate(n); out
}
fn bad(re: &Regex, t: &str) -> bool {
    spans(t) != re.find_iter(t).map(|(a,b)| Span::new(a as u32,b as u32)).collect::<Vec<_>>()
}
#[test]
#[ignore]
fn probe() {
    let re = Regex::new(O200K).unwrap();
    const ALPHA: &[&str] = &["a","b","z","A","Q","\u{e9}","\u{df}","\u{132}","0","1","9"," ","  ",
        "\n","\r\n","\t","'","'s","'ll",".","!","/","#","_","\u{4e2d}","\u{6587}","\u{3072}",
        "\u{30ab}","\u{d55c}","\u{645}","\u{1f600}","\u{301}","\u{200d}","\u{bd}","\u{2168}"];
    let mut rs = 0x243F_6A88_85A3_08D3u64;
    let mut next = move || { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; rs };
    let mut shown = 0;
    for _ in 0..2000 {
        let len = 1 + (next() % 300) as usize;
        let t: String = (0..len).map(|_| ALPHA[(next() % ALPHA.len() as u64) as usize]).collect();
        if !bad(&re, &t) { continue; }
        // shortest failing prefix
        let mut lo = 0usize; let mut hi = t.len();
        while lo < hi {
            let mut m = (lo + hi) / 2;
            while m < t.len() && !t.is_char_boundary(m) { m += 1; }
            if m >= t.len() { break; }
            if bad(&re, &t[..m]) { hi = m; } else { lo = m + 1; }
        }
        let mut p = &t[..hi.min(t.len())];
        // then trim from the left as far as it stays failing
        let mut best = p.to_string();
        for i in 1..best.len() {
            if best.is_char_boundary(i) && bad(&re, &best[i..]) { p = ""; best = best[i..].to_string(); break; }
        }
        let _ = p;
        let g = spans(&best);
        let wv: Vec<Span> = re.find_iter(&best).map(|(a,b)| Span::new(a as u32,b as u32)).collect();
        println!("MIN len={} {:?}", best.len(), best);
        println!("  got ={:?}", g.iter().map(|s| &best[s.range()]).collect::<Vec<_>>());
        println!("  want={:?}", wv.iter().map(|s| &best[s.range()]).collect::<Vec<_>>());
        shown += 1;
        if shown >= 2 { return; }
    }
    if shown == 0 { println!("no fuzz failure"); }
}
