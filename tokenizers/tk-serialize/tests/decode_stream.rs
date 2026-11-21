//! Tests [`DecodeStream`] on real tokenizer configs. The expected values are what the
//! released 0.23.3 `DecodeStream` returns for the same ids.

use tk_encode::pipeline::{DecodeStream, Tokenizer};

fn load(name: &str) -> Tokenizer {
    let canonical = tk_convert::canonicalize_file(format!("../data/{name}")).unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

/// A byte-fallback tokenizer built inline, so the test needs no fixture files.
fn byte_fallback() -> (Tokenizer, Vec<&'static str>) {
    let mut vocab = String::from("\"<unk>\":0,\"a\":1,\"ok\":2");
    for byte in 0..=255u8 {
        vocab.push_str(&format!(",\"<0x{byte:02X}>\":{}", byte as usize + 3));
    }
    let json = format!(
        r#"{{"version":"2.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,
        "decoder":{{"type":"ByteFallback"}},
        "model":{{"type":"WordLevel","vocab":{{{vocab}}},"unk_token":"<unk>"}}}}"#
    );
    let tokenizer = tk_serialize::from_json(&json).unwrap();
    (
        tokenizer,
        vec!["a", "<0xF0>", "<0xA0>", "<0x80>", "<0x80>", "<0xE4>", "ok"],
    )
}

#[test]
fn holds_back_a_character_until_its_last_byte() {
    let llama3 = load("llama-3-tokenizer.json");
    let mut stream = DecodeStream::new(Vec::new(), false);

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
    let mut stream = DecodeStream::new(vec![bos, hello], true);
    assert_eq!(
        stream.step(&llama2, &[world]).unwrap(),
        Some(" world".into())
    );
    let mut stream = DecodeStream::new(Vec::new(), true);
    assert_eq!(
        stream.step(&llama2, &[world]).unwrap(),
        Some("world".into())
    );
}

#[test]
fn a_stray_byte_does_not_rewrite_text_the_stream_already_returned() {
    let llama2 = load("llama-2.json");
    let mut stream = DecodeStream::new(Vec::new(), false);

    // <0x61> decodes to "a". <0xD5> starts a character, but the next token is not a byte,
    // so the run becomes a replacement character.
    //
    // This sequence used to raise "Invalid prefix encountered": the decoder turned the
    // pending run into one U+FFFD per byte, the step that added <0xD5> returned none, and
    // the next decode started with two replacement characters where the stream had
    // recorded one. The decoder now replaces only the byte that cannot be decoded, so the
    // stream keeps the prefix it already returned and continues.
    assert_eq!(stream.step(&llama2, &[100]).unwrap(), Some("a".into()));
    assert_eq!(stream.step(&llama2, &[216]).unwrap(), None);
    assert_eq!(stream.step(&llama2, &[31479]).unwrap(), Some("�写".into()));
}

#[cfg(feature = "wordlevel")]
#[test]
fn byte_fallback_stream_keeps_a_valid_character_before_a_stray_byte() {
    let (tokenizer, tokens) = byte_fallback();
    let ids: Vec<u32> = tokens
        .iter()
        .map(|t| tokenizer.token_to_id(t).unwrap())
        .collect();

    // F0 A0 80 80 is U+20000, E4 is a lead byte with no continuation. The run used
    // to decode to one replacement per byte, which both dropped the valid
    // character and made the stream reject its own earlier output.
    let mut stream = DecodeStream::new(Vec::new(), false);
    let mut text = String::new();
    for id in ids {
        if let Some(chunk) = stream.step(&tokenizer, &[id]).unwrap() {
            text.push_str(&chunk);
        }
    }
    assert_eq!(text, "a𠀀�ok");
}
