//! Encode every corpus with a `.tok` and print a digest per corpus.
//!
//! Runs with or without the `config` feature, so the two builds can be diffed against each other:
//! the read-only build must produce exactly what the full one does.

use tk_encode::pipeline::PipelineTokenizer;

const CORPORA: &[&str] = &[
    "english", "chinese", "code", "dense", "russian", "arabic", "korean", "greek", "hindi", "thai",
];

fn main() {
    for path in std::env::args().skip(1) {
        let file = tk_serialization::TokFile::open(&path).expect("open .tok");
        let tok = PipelineTokenizer::from_tok(file.bytes()).expect("load .tok");
        for name in CORPORA {
            let Ok(text) = std::fs::read_to_string(format!("data/corpora/{name}.txt")) else {
                continue;
            };
            let ids = tok.encode(text.as_str(), true).expect("encode");
            // FNV-1a over the ids: a mismatch anywhere changes it.
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for token in &ids {
                for b in token.id.to_le_bytes() {
                    h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
                }
            }
            println!("{path} {name} {} {h:016x}", ids.len());
        }
    }
}
