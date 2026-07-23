//! Aligned per-step ns/B, tk-encode (PR #2190 poc-merge-cache) vs gigatoken, one process.
//! Verifies token-id equality first. run: cargo +nightly run --release
//!
//! ours steps via the pipeline stage ladder (encode_generic::<STAGE>): marginals
//!   normalize = t(NORMALIZE)-t(FRAME), split = t(SPLIT)-t(NORMALIZE),
//!   model = t(MODEL)-t(SPLIT) (warm = FlatCache hot; cold = fresh instance).
//! theirs steps via public API: split = pretokenize().count(),
//!   probe = memoized_encode_flat warm - split, merge = encode cold - warm.
use std::hint::black_box;
use std::time::Instant;

use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;
use tk_encode::tokenizer::pipeline::{PipelineToken, PipelineTokenizer};
use tk_encode::tokenizer::Tokenizer;

const ROOT: &str = "/Users/arthurzucker/Work/tokenizers/tokenizers";

fn best(len: usize, iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..2 {
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

fn corpus(rel: &str) -> Option<String> {
    let s = std::fs::read_to_string(format!("{ROOT}/{rel}")).ok()?;
    let mut c = s.len().min(1_000_000);
    while c > 0 && !s.is_char_boundary(c) {
        c -= 1;
    }
    Some(s[..c].to_string())
}

// one stage-ladder level, best-of-N (warm cache after warmup)
fn ours_stage<const STAGE: u8>(pt: &PipelineTokenizer, text: &str, n: usize, it: u32) -> f64 {
    best(n, it, || {
        let mut out: Vec<PipelineToken> = Vec::new();
        let mut pre = Vec::new();
        pt.encode_generic::<STAGE>(black_box(text), &mut out, &mut pre).unwrap();
        black_box(out.len());
    })
}

fn main() {
    let toks: &[(&str, &str)] = &[
        ("gpt2", "data/gpt2.json"),
        ("deepseek-v4", "data/deepseek-v4.json"),
        ("llama-3", "data/llama-3-tokenizer.json"),
    ];
    let corpora: &[(&str, &str)] = &[
        ("English", "data/big.txt"),
        ("Russian", "atomsplit/benches/data/ru.txt"),
        ("Chinese", "atomsplit/benches/data/zh.txt"),
        ("Korean", "atomsplit/benches/data/ko.txt"),
    ];

    println!(
        "\nns/byte per step. OURS = tk-encode #2190 (pipeline stage ladder). THEIRS = gigatoken.\n\
         norm/split/model marginals; model warm = cache-hit+emit, cold = full BPE merge.\n\
         theirs: split=pretokenize, probe=warm memoized-split, merge=cold-warm. idsEq gates trust.\n"
    );
    println!(
        "{:<11} {:<8} {:>7} | {:>5} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>5}",
        "tok", "lang", "bytes",
        "o.nrm", "o.splt", "o.mdlW", "o.mdlC", "o.tot",
        "t.splt", "t.prb", "t.mrg", "idsEq"
    );

    for (tname, tpath) in toks {
        let path = format!("{ROOT}/{tpath}");
        let base = match Tokenizer::from_file(&path) {
            Ok(t) => t,
            Err(e) => { eprintln!("skip {tname}: ours load {e}"); continue; }
        };
        let ours = match PipelineTokenizer::try_from(&base) {
            Ok(p) => p,
            Err(e) => { eprintln!("skip {tname}: ours pipeline {e:?}"); continue; }
        };
        if load_hf_bpe(&path).is_err() {
            eprintln!("skip {tname}: giga load"); continue;
        }

        for (label, rel) in corpora {
            let Some(text) = corpus(rel) else { continue };
            let n = text.len();
            let it = (20_000_000 / n).clamp(2, 40) as u32;

            // --- id equality (warm both) ---
            let o_ids: Vec<u32> = ours.encode(&text, false).unwrap().into_iter().map(|t| t.id).collect();
            let mut giga = load_hf_bpe(&path).unwrap();
            let g_ids = { let mut v = Vec::new(); giga.encode_with_added_tokens_flat(text.as_bytes(), &mut v); v };
            let eq = o_ids == g_ids;
            let idseq = if eq { "OK".into() } else {
                format!("x{}/{}", o_ids.len(), g_ids.len())
            };

            // --- OURS stage ladder (warm: FlatCache hot in `ours`) ---
            let t_frame = ours_stage::<0>(&ours, &text, n, it);
            let t_norm  = ours_stage::<1>(&ours, &text, n, it);
            let t_split = ours_stage::<2>(&ours, &text, n, it);
            let t_modelw= ours_stage::<3>(&ours, &text, n, it);
            let o_nrm = (t_norm - t_frame).max(0.0);
            let o_splt = (t_split - t_norm).max(0.0);
            let o_mdlw = (t_modelw - t_split).max(0.0);
            // ours cold model: fresh instance, one-shot full encode minus split
            let o_mdlc = {
                let fresh = PipelineTokenizer::try_from(&base).unwrap();
                let mut out: Vec<PipelineToken> = Vec::new();
                let mut pre = Vec::new();
                let t0 = Instant::now();
                fresh.encode_generic::<3>(&text, &mut out, &mut pre).unwrap();
                (t0.elapsed().as_nanos() as f64 / n as f64 - t_split).max(0.0)
            };
            let o_tot = t_modelw;

            // --- THEIRS steps (pt is a Copy enum; does not borrow giga) ---
            let pt = giga.pretokenizer_type();
            let t_tsplit = best(n, it, || {
                black_box(pt.pretokenize(black_box(text.as_bytes())).count());
            });
            let mut scratch = Vec::new();
            giga.memoized_encode_flat(pt.pretokenize(text.as_bytes()), &mut scratch); // warm
            let t_tmemo = best(n, it, || {
                scratch.clear();
                giga.memoized_encode_flat(pt.pretokenize(black_box(text.as_bytes())), &mut scratch);
                black_box(scratch.len());
            });
            let t_prb = (t_tmemo - t_tsplit).max(0.0);
            let t_warm = best(n, it, || {
                scratch.clear();
                giga.encode_with_added_tokens_flat(black_box(text.as_bytes()), &mut scratch);
                black_box(scratch.len());
            });
            let t_cold = {
                let mut f = load_hf_bpe(&path).unwrap();
                let mut v = Vec::new();
                let t0 = Instant::now();
                f.encode_with_added_tokens_flat(text.as_bytes(), &mut v);
                t0.elapsed().as_nanos() as f64 / n as f64
            };
            let t_mrg = (t_cold - t_warm).max(0.0);

            println!(
                "{tname:<11} {label:<8} {n:>7} | {o_nrm:>5.2} {o_splt:>6.2} {o_mdlw:>6.2} {o_mdlc:>6.2} {o_tot:>6.2} | {t_tsplit:>6.2} {t_prb:>6.2} {t_mrg:>6.2} | {idseq:>5}"
            );
        }
        println!();
    }
}
