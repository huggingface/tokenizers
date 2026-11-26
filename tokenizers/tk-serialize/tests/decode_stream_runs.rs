//! Streaming runs: a delimiter arrives as an id, not as text to be matched.

use tk_encode::pipeline::DecodeRuns;
use tk_encode::pipeline::DecodeStream;
use tk_encode::pipeline::Decoded;
use tk_encode::pipeline::EncodeOptions;
use tk_encode::pipeline::Tokenizer;

/// `data/gpt2.json` is still version `1.0`, so run the upgrade pass first -- the reader only
/// accepts canonical `2.0`. Same as `benches/encode.rs`.
fn load() -> Tokenizer {
    let canonical = tk_convert::canonicalize_file("../data/gpt2.json").unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

const EOT: u32 = 50256;
const MARKER: &str = "<|endoftext|>";

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, &EncodeOptions::default())
        .wait()
        .unwrap()
        .into_iter()
        .flat_map(|encoding| encoding.ids().iter().map(|token| token.id()).collect::<Vec<u32>>())
        .collect()
}

/// Runs out, one within at most two ids, the way an id at a time arrives.
fn step(tokenizer: &Tokenizer, prompt: Vec<u32>, ids: &[u32]) -> (String, usize, Vec<String>) {
    let mut stream = DecodeStream::new(prompt, false);
    let mut runs = DecodeRuns::default();
    let (mut text, mut delimiters, mut added) = (String::new(), 0, Vec::new());
    for pair in ids.chunks(2) {
        if stream.step_runs(tokenizer, pair, &mut runs).unwrap() {
            text.push_str(runs.text());
            for run in runs.iter() {
                if let Decoded::Added { text, .. } = run {
                    delimiters += 1;
                    added.push(text.to_string());
                }
            }
        }
    }
    (text, delimiters, added)
}

#[test]
fn the_runs_of_a_stream_are_the_runs_of_the_whole_decode() {
    let tokenizer = load();
    let text = format!("a{MARKER}b plain {MARKER} text");
    let ids = encode(&tokenizer, &text);

    let (streamed, delimiters, added) = step(&tokenizer, Vec::new(), &ids);
    let mut runs = DecodeRuns::default();
    tokenizer.decode_runs(&ids, false, &mut runs).unwrap();

    assert_eq!(streamed, runs.text(), "the stream spells what the whole decode spells");
    assert_eq!(streamed, text);
    assert_eq!(delimiters, 2);
    assert_eq!(added, [MARKER, MARKER]);

    let whole = runs
        .iter()
        .filter(|run| matches!(run, Decoded::Added { .. }))
        .count();
    assert_eq!(delimiters, whole, "and finds the same delimiters");
}

#[test]
fn a_marker_spelled_out_is_text_on_both_paths() {
    let tokenizer = load();
    let mut options = EncodeOptions::default();
    options.encode_special_tokens = true;
    let ids: Vec<u32> = tokenizer
        .encode(MARKER, &options)
        .wait()
        .unwrap()
        .into_iter()
        .flat_map(|encoding| encoding.ids().iter().map(|token| token.id()).collect::<Vec<u32>>())
        .collect();
    assert!(!ids.contains(&EOT));

    let (streamed, delimiters, _) = step(&tokenizer, Vec::new(), &ids);
    assert_eq!(streamed, MARKER);
    assert_eq!(delimiters, 0, "a spelling is not a delimiter, streamed or not");
}

#[test]
fn the_prompt_is_context_and_never_emitted() {
    let tokenizer = load();
    let prompt = encode(&tokenizer, &format!("prompt {MARKER} "));
    let ids = encode(&tokenizer, "the answer");

    let (streamed, delimiters, _) = step(&tokenizer, prompt, &ids);
    assert_eq!(streamed, "the answer");
    assert_eq!(delimiters, 0, "nothing the prompt contains is emitted");
}
