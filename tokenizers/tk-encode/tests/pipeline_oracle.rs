//! The experimental `PipelineTokenizer` must produce exactly the same token
//! ids as the reference `Tokenizer` (the oracle). Exercised over two tokenizer
//! configs — `bert-base-uncased` (`BertNormalizer` + `BertPreTokenizer` +
//! `WordPiece`) and `llama-2` (SentencePiece-style: `Prepend`/`Replace`
//! normalizer, *no* pre-tokenizer, byte-fallback `BPE`) — against a ~1 MB
//! fraction of several language fixtures, with lines packed into ~1 kB and
//! ~10 kB documents. The oracle is called with `add_special_tokens = false`
//! because the pipeline does not apply the post-processor yet.

use std::convert::TryFrom;

use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

/// Cap each language fixture to a ~1 MB, line-bounded prefix so the oracle
/// stays affordable while still exercising real multilingual text.
const FRACTION_BYTES: usize = 1024 * 1024;

fn load_tokenizers(config: &str) -> (Tokenizer, PipelineTokenizer) {
    let oracle = Tokenizer::from_file(config).unwrap();
    let pipeline = PipelineTokenizer::try_from(&oracle).unwrap();
    (oracle, pipeline)
}

/// Read `corpus`, keeping whole lines until adding the next would exceed
/// `max_bytes`. Whole lines keep the prefix valid UTF-8.
fn load_fraction(corpus: &str, max_bytes: usize) -> String {
    let text = std::fs::read_to_string(corpus).unwrap();
    let mut out = String::new();
    for line in text.lines() {
        if !out.is_empty() && out.len() + line.len() + 1 > max_bytes {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
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

fn check_chunks(config: &str, corpus: &str, target_bytes: usize) {
    let (oracle, pipeline) = load_tokenizers(config);
    let text = load_fraction(corpus, FRACTION_BYTES);
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

/// One `chunks_1kb` / `chunks_10kb` case per corpus, all against a single
/// tokenizer `$config`. Invoke inside a `mod <tokenizer> { .. }` so cases nest
/// as `<tokenizer>::<corpus>`.
macro_rules! corpus_tests {
    ($config:literal; $($corpus:ident => $corpus_path:literal),* $(,)?) => {
        $(
            mod $corpus {
                #[test]
                fn chunks_1kb() {
                    crate::check_chunks($config, $corpus_path, 1024);
                }
                #[test]
                fn chunks_10kb() {
                    crate::check_chunks($config, $corpus_path, 10 * 1024);
                }
            }
        )*
    };
}

macro_rules! lang_fixtures {
    ($config:literal) => {
        corpus_tests! { $config;
            eng_latn => "../data/fixtures/lang/eng_Latn.txt",
            rus_cyrl => "../data/fixtures/lang/rus_Cyrl.txt",
            arb_arab => "../data/fixtures/lang/arb_Arab.txt",
            cmn_hani => "../data/fixtures/lang/cmn_Hani.txt",
            jpn_jpan => "../data/fixtures/lang/jpn_Jpan.txt",
            hin_deva => "../data/fixtures/lang/hin_Deva.txt",
            kor_hang => "../data/fixtures/lang/kor_Hang.txt",
            tha_thai => "../data/fixtures/lang/tha_Thai.txt",
        }
    };
}

mod bert_base_uncased {
    lang_fixtures!("../data/bert-base-uncased.json");
}

mod llama2 {
    lang_fixtures!("../data/llama-2-7b-chat-hf.json");
}
