//! The experimental `PipelineTokenizer` must produce exactly the same token
//! ids as the reference `Tokenizer` (the oracle). Exercised over the bert-wiki
//! tokenizer (`Whitespace` pre-tokenizer + `WordPiece`) and the t5-base
//! tokenizer (`WhitespaceSplit` + `Metaspace` rewriter + `Unigram`) on an
//! English and a Japanese corpus, with lines packed into ~1 kB and ~10 kB
//! documents. The oracle is called with `add_special_tokens = false` because
//! the pipeline does not apply the post-processor yet.

use std::convert::TryFrom;

use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

fn load(tokenizer: &str, corpus: &str) -> (Tokenizer, PipelineTokenizer, String) {
    let oracle = Tokenizer::from_file(tokenizer).unwrap();
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

fn check_chunks(tokenizer: &str, corpus: &str, target_bytes: usize) {
    let (oracle, pipeline, text) = load(tokenizer, corpus);
    for chunk in make_chunks(&text, target_bytes) {
        let expected = oracle.encode(chunk.as_str(), false).unwrap();
        let got: Vec<u32> = pipeline
            .encode(&chunk, false)
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        assert_eq!(
            expected.get_ids(),
            got.as_slice(),
            "id mismatch on {:?}",
            chunk.chars().take(80).collect::<String>(),
        );
    }
}

macro_rules! corpus_tests {
    ($($name:ident => ($tokenizer:literal, $corpus:literal)),* $(,)?) => {
        $(
            mod $name {
                #[test]
                fn chunks_1kb() {
                    super::check_chunks($tokenizer, $corpus, 1024);
                }
                #[test]
                fn chunks_10kb() {
                    super::check_chunks($tokenizer, $corpus, 10 * 1024);
                }
            }
        )*
    };
}

corpus_tests! {
    bert_wiki_big => ("../data/bert-wiki.json", "../data/big.txt"),
    bert_wiki_wagahai => ("../data/bert-wiki.json", "../data/unigram_wagahaiwa_nekodearu.txt"),
    t5_big => ("../data/t5-base.json", "../data/big.txt"),
    t5_wagahai => ("../data/t5-base.json", "../data/unigram_wagahaiwa_nekodearu.txt"),
}
