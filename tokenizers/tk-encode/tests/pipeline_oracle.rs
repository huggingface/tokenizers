//! The experimental `PipelineTokenizer` must produce exactly the same token
//! ids as the reference `Tokenizer` (the oracle). Exercised over the bert-wiki
//! tokenizer (`Whitespace` pre-tokenizer + `WordPiece`) on an English and a
//! Japanese corpus, with lines packed into ~1 kB and ~10 kB documents, for both
//! `add_special_tokens` values. bert-wiki's post-processor is
//! `Template("[CLS] A [SEP]")`, so `true` genuinely exercises `STAGE_POSTPROCESS`
//! on both corpora.

use std::convert::TryFrom;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

fn load(corpus: &str) -> (Tokenizer, PipelineTokenizer, String) {
    let oracle = Tokenizer::from_file("../data/bert-wiki.json").unwrap();
    let pipeline = PipelineTokenizer::try_from(&oracle).unwrap();
    let text = std::fs::read_to_string(corpus).unwrap();
    (oracle, pipeline, text)
}

fn make_chunks(text: &str, target_bytes: usize) -> Vec<String> {
    let lines = text.lines().filter(|l| !l.trim().is_empty());
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in lines {
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
        if cur.len() >= target_bytes {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

fn check_chunks(corpus: &str, target_bytes: usize) {
    let (oracle, pipeline, text) = load(corpus);
    for chunk in make_chunks(&text, target_bytes) {
        for add_special_tokens in [false, true] {
            let expected = oracle.encode(chunk.as_str(), add_special_tokens).unwrap();
            let got: Vec<u32> = pipeline
                .encode(&chunk, add_special_tokens)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect();
            assert_eq!(
                expected.get_ids(),
                got.as_slice(),
                "id mismatch (add_special_tokens={add_special_tokens}) on {:?}",
                chunk.chars().take(80).collect::<String>(),
            );
        }
    }
}

macro_rules! corpus_tests {
    ($($name:ident => $file:literal),* $(,)?) => {
        $(
            mod $name {
                #[test]
                fn chunks_1kb() {
                    super::check_chunks($file, 1024);
                }
                #[test]
                fn chunks_10kb() {
                    super::check_chunks($file, 10 * 1024);
                }
            }
        )*
    };
}

corpus_tests! {
    big => "../data/big.txt",
    wagahai => "../data/unigram_wagahaiwa_nekodearu.txt",
}
