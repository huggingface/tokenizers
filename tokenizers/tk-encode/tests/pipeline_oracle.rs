//! The experimental `PipelineTokenizer` must produce exactly the same token
//! ids as the reference `Tokenizer` (the oracle). Exercised over the bert-wiki
//! tokenizer (`Whitespace` pre-tokenizer + `WordPiece`) on an English and a
//! Japanese corpus, with lines packed into ~1 kB and ~10 kB documents. The
//! oracle is called with `add_special_tokens = false` because the pipeline
//! does not apply the post-processor yet.

use std::convert::TryFrom;

use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

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
        let expected = oracle.encode(chunk.as_str(), false).unwrap();
        let got: Vec<u32> = pipeline
            .encode(&chunk).into_single()
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

/// Batch parallel floor: a multi-sequence `encode` (fanned out on the pool) must
/// be result-identical to both the serial pipeline `encode` (order-preserving
/// parallelism) and the reference oracle. Also checks the two `EncodeHandle`
/// consumers agree: `wait_for_completion` (bulk collect) and `Iterator`
/// (input-ordered streaming). Runs the whole corpus as one batch so multiple
/// documents land on different worker threads.
fn check_batch(corpus: &str, target_bytes: usize) {
    let (oracle, pipeline, text) = load(corpus);
    let chunks = make_chunks(&text, target_bytes);
    let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();

    let batch = pipeline
        .encode(&refs[..]).wait_for_completion()
        .unwrap();
    assert_eq!(batch.len(), chunks.len(), "batch length mismatch");

    // Iterator face must yield the same results, in input order.
    let streamed: Vec<Vec<u32>> = pipeline
        .encode(&refs[..])
        .map(|r| r.unwrap().iter().map(|t| t.id).collect())
        .collect();

    for (i, (chunk, got)) in chunks.iter().zip(&batch).enumerate() {
        let got_ids: Vec<u32> = got.iter().map(|t| t.id).collect();

        // wait_for_completion == Iterator
        assert_eq!(got_ids, streamed[i], "wait_for_completion != Iterator");

        // batch == serial pipeline
        let serial: Vec<u32> = pipeline
            .encode(chunk).into_single()
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        assert_eq!(
            got_ids, serial,
            "batch != serial on {:?}",
            chunk.chars().take(80).collect::<String>(),
        );

        // batch == reference oracle
        let expected = oracle.encode(chunk.as_str(), false).unwrap();
        assert_eq!(
            expected.get_ids(),
            got_ids.as_slice(),
            "batch != oracle on {:?}",
            chunk.chars().take(80).collect::<String>(),
        );
    }
}

/// Exercises the cost gate's *inline* branch: a multi-sequence `encode` whose
/// summed size is below the fan-out threshold runs on the calling thread (not
/// the pool) and must still match the oracle. The corpus `check_batch` always
/// clears the threshold and fans out, so this covers the other side.
fn check_small_batch(corpus: &str) {
    let (oracle, pipeline, text) = load(corpus);
    // A handful of short lines — well under the fan-out byte threshold.
    let inputs: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(4)
        .collect();
    if inputs.len() < 2 {
        return;
    }
    let got = pipeline
        .encode(&inputs[..]).wait_for_completion()
        .unwrap();
    assert_eq!(got.len(), inputs.len());
    for (input, tokens) in inputs.iter().zip(&got) {
        let expected = oracle.encode(*input, false).unwrap();
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        assert_eq!(
            expected.get_ids(),
            ids.as_slice(),
            "small-batch (inline) mismatch on {input:?}",
        );
    }
}

/// Intra-sequence chunking: a single large document is encoded through the fused stride
/// path — split at newline boundaries into chunks, each run through the full pipeline in
/// parallel, and concatenated in order. The result must equal the oracle (which, combined
/// with the serial `check_chunks` above, transitively confirms chunked == serial == oracle).
/// The bert-wiki config is chunk-safe (per-char normalizer + whitespace-delimiting
/// pre-tokenizer), so this exercises the split; an unsafe config would stay whole.
fn check_intra_seq(corpus: &str) {
    let (oracle, pipeline, text) = load(corpus);
    // One document large enough to be split into several chunks, but capped so the oracle
    // reference encode stays cheap.
    let mut doc = String::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if !doc.is_empty() {
            doc.push('\n');
        }
        doc.push_str(line);
        if doc.len() >= 64 * 1024 {
            break;
        }
    }
    if doc.len() < 16 * 1024 {
        return; // corpus too small to exercise the parallel path
    }
    let got: Vec<u32> = pipeline
        .encode(doc.as_str()).into_single()
        .unwrap()
        .iter()
        .map(|t| t.id)
        .collect();
    let expected = oracle.encode(doc.as_str(), false).unwrap();
    assert_eq!(
        expected.get_ids(),
        got.as_slice(),
        "intra-seq parallel encode != oracle",
    );
}

/// Byte-level BPE (llama-3): the `SpaceRun` raw-cut path must be id-identical
/// to the reference oracle across large real documents. Exercised over English
/// (dense with non-ws→space cut candidates) and Japanese (multibyte seams —
/// cuts may only land beside ASCII), through both encode paths.
fn check_llama3(corpus: &str) {
    let oracle = Tokenizer::from_file("../data/llama-3-tokenizer.json").unwrap();
    let pipeline = PipelineTokenizer::try_from(&oracle).unwrap();
    let text = std::fs::read_to_string(corpus).unwrap();
    let mut doc = String::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if !doc.is_empty() {
            doc.push('\n');
        }
        doc.push_str(line);
        if doc.len() >= 256 * 1024 {
            break;
        }
    }
    if doc.len() < 32 * 1024 {
        return;
    }
    let expected = oracle.encode(doc.as_str(), false).unwrap();
    let single: Vec<u32> = pipeline
        .encode(doc.as_str()).into_single()
        .unwrap()
        .iter()
        .map(|t| t.id)
        .collect();
    assert_eq!(
        expected.get_ids(),
        single.as_slice(),
        "llama-3 single-doc SpaceRun encode != oracle"
    );
    let owned: Vec<u32> = pipeline
        .encode(doc)
        .into_single()
        .unwrap()
        .iter()
        .map(|t| t.id)
        .collect();
    assert_eq!(
        expected.get_ids(),
        owned.as_slice(),
        "llama-3 owned SpaceRun encode != oracle"
    );
}

#[test]
fn llama3_intra_seq_english() {
    check_llama3("../data/big.txt");
}

#[test]
fn llama3_intra_seq_japanese() {
    check_llama3("../data/unigram_wagahaiwa_nekodearu.txt");
}

/// Byte-level BPE on a whitespace-free document (base64-like: letters + digits,
/// no spaces or newlines) must stay byte-identical to the oracle through the
/// number-transition cuts — the case that previously degraded to whole-input
/// serial because `boundary_fsm` found no space or newline to cut at.
#[test]
fn llama3_intra_seq_whitespace_free() {
    let oracle = Tokenizer::from_file("../data/llama-3-tokenizer.json").unwrap();
    let pipeline = PipelineTokenizer::try_from(&oracle).unwrap();
    // ~200 KB, no whitespace, dense letter↔digit flips (the cut sites).
    let doc = "aGVsbG8x3Wb29ybGQ7abc123def456ghi789jkl0mno".repeat(5000);
    assert!(!doc.bytes().any(|b| b.is_ascii_whitespace()));
    let expected = oracle.encode(doc.as_str(), false).unwrap();
    let single: Vec<u32> = pipeline
        .encode(doc.as_str())
        .into_single()
        .unwrap()
        .iter()
        .map(|t| t.id)
        .collect();
    assert_eq!(
        expected.get_ids(),
        single.as_slice(),
        "whitespace-free byte-level encode != oracle"
    );
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
                fn intra_seq() {
                    super::check_intra_seq($file);
                }
                #[test]
                fn chunks_10kb() {
                    super::check_chunks($file, 10 * 1024);
                }
                #[test]
                fn batch_1kb() {
                    super::check_batch($file, 1024);
                }
                #[test]
                fn batch_10kb() {
                    super::check_batch($file, 10 * 1024);
                }
                #[test]
                fn small_batch_inline() {
                    super::check_small_batch($file);
                }
            }
        )*
    };
}

corpus_tests! {
    big => "../data/big.txt",
    wagahai => "../data/unigram_wagahaiwa_nekodearu.txt",
}
