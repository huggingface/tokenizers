//! Tests incremental decoding and prefix handling in [`PipelineDecodeStream`].

use tk_encode::pipeline::{PipelineDecodeStream, PipelineTokenizer};

fn load(name: &str) -> PipelineTokenizer {
    let canonical = tk_convert::canonicalize_file(format!("../data/{name}")).unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

#[test]
fn holds_back_a_character_until_its_last_byte() {
    let llama3 = load("llama-3-tokenizer.json");
    let mut stream = PipelineDecodeStream::new(Vec::new(), false);

    // "삥뽕빵": each syllable takes two or three byte-level tokens
    let steps: Vec<_> = [80690, 98, 167, 121, 243]
        .iter()
        .map(|&id| stream.step(&llama3, &[id]).unwrap())
        .collect();
    assert_eq!(
        steps,
        [None, Some("삥".into()), None, None, Some("뽕".into())]
    );
    assert_eq!(
        stream.step(&llama3, &[102457, 113]).unwrap(),
        Some("빵".into())
    );
}

#[test]
fn prompt_ids_are_context_not_output() {
    let llama2 = load("llama-2.json");
    // "<s>", "▁Hello", "▁world"
    let (bos, hello, world) = (1, 15043, 3186);

    // The Metaspace decoder strips the space in front of the first word only
    let mut stream = PipelineDecodeStream::new(vec![bos, hello], true);
    assert_eq!(
        stream.step(&llama2, &[world]).unwrap(),
        Some(" world".into())
    );
    let mut stream = PipelineDecodeStream::new(Vec::new(), true);
    assert_eq!(
        stream.step(&llama2, &[world]).unwrap(),
        Some("world".into())
    );
}

#[test]
fn byte_fallback_preserves_the_emitted_prefix() {
    let llama2 = load("llama-2.json");
    let mut stream = PipelineDecodeStream::new(Vec::new(), false);

    // <0x61> decodes to "a". <0xD5> starts a character, but the next token is not a byte.
    // Only the incomplete byte is replaced.
    assert_eq!(stream.step(&llama2, &[100]).unwrap(), Some("a".into()));
    assert_eq!(stream.step(&llama2, &[216]).unwrap(), None);
    assert_eq!(stream.step(&llama2, &[31479]).unwrap(), Some("�写".into()));
}

#[test]
fn a_rewritten_prefix_is_an_invalid_prefix_error() {
    // Fusing "a" and "b" makes the replacement match across the token boundary.
    let tokenizer = tk_serialize::from_json(
        r#"{
        "version": "2.0", "model": {"type": "BPE", "byte_level": false,
            "vocab": {"a": 0, "b": 1}, "merges": []},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Fuse"}, {"type": "Replace", "pattern": {"String": "ab"}, "content": "XY"}
        ]}
    }"#,
    )
    .unwrap();
    let mut stream = PipelineDecodeStream::new(Vec::new(), false);
    assert_eq!(stream.step(&tokenizer, &[0]).unwrap(), Some("a".into()));
    let error = stream.step(&tokenizer, &[1]).unwrap_err().to_string();
    // vLLM resets its stream when an error message starts with these words
    // (vllm/v1/engine/detokenizer.py).
    assert!(error.starts_with("Invalid prefix encountered"), "{error}");
}
