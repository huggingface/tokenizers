//! [`PipelineDecodeStream`] over real configs. Expected values are what released 0.23.3's
//! `DecodeStream` returns for the same ids.

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
fn a_rewritten_prefix_is_an_invalid_prefix_error() {
    let llama2 = load("llama-2.json");
    let mut stream = PipelineDecodeStream::new(Vec::new(), false);

    // <0x61> decodes to "a", then <0xD5> with no continuation byte turns both into U+FFFD
    assert_eq!(stream.step(&llama2, &[100]).unwrap(), Some("a".into()));
    assert_eq!(stream.step(&llama2, &[216]).unwrap(), None);
    let error = stream.step(&llama2, &[31479]).unwrap_err().to_string();
    // vLLM matches this prefix to reset its stream (vllm/v1/engine/detokenizer.py)
    assert!(error.starts_with("Invalid prefix encountered"), "{error}");
}
