#![cfg(feature = "deserialize")]

use serde_json::{Value, json};
use tk_encode::pipeline::{EncodeOptions, PipelineTokenizer};
use tk_encode::utils::byte_level::BYTES_CHAR_LOOKUP;

fn tokenizer(decoder: Value, normalize_newlines: bool) -> PipelineTokenizer {
    tk_encode::utils::parallelism::set_num_threads(2);
    let vocab: serde_json::Map<String, Value> = BYTES_CHAR_LOOKUP
        .iter()
        .enumerate()
        .map(|(byte, c)| (c.to_string(), json!(byte)))
        .collect();
    let normalizer = if normalize_newlines {
        json!({"type": "Replace", "pattern": {"String": "\n"}, "content": "<NL>"})
    } else {
        Value::Null
    };
    tk_serialize::from_json(
        &json!({
            "version": "2.0",
            "normalizer": normalizer,
            "decoder": decoder,
            "added_tokens": [
                {"id": 256, "content": "<s>", "special": true, "normalized": false,
                 "single_word": false, "lstrip": false, "rstrip": false},
                {"id": 257, "content": "<tag>", "special": false, "normalized": false,
                 "single_word": false, "lstrip": false, "rstrip": false}
            ],
            "model": {"type": "BPE", "byte_level": true, "vocab": vocab, "merges": []}
        })
        .to_string(),
    )
    .unwrap()
}

fn ids(tokenizer: &PipelineTokenizer, input: &str) -> Vec<u32> {
    tokenizer
        .encode(input, &EncodeOptions::no_specials())
        .wait()
        .unwrap()[0]
        .ids()
        .iter()
        .map(|t| t.id())
        .collect()
}

#[test]
fn byte_level_decoder_sequence_reverses_normalization() {
    let tokenizer = tokenizer(
        json!({"type": "Sequence", "decoders": [
            {"type": "ByteLevel"},
            {"type": "Replace", "pattern": {"String": "<NL>"}, "content": "\n"}
        ]}),
        true,
    );
    for input in ["Hello\n世界", "\n", "a\n\nb", "", "é Ā 😀"] {
        let ids = ids(&tokenizer, input);
        assert_eq!(
            ids,
            input
                .replace('\n', "<NL>")
                .bytes()
                .map(u32::from)
                .collect::<Vec<_>>()
        );
        assert_eq!(tokenizer.decode(&ids, true).unwrap(), input);
        assert_eq!(tokenizer.decode_batch(&[&ids], true).unwrap(), [input]);
    }
}

#[test]
fn byte_level_decoder_sequence_preserves_order() {
    let replace = json!({"type": "Replace", "pattern": {"String": "Ġ"}, "content": "_"});
    let byte_level = json!({"type": "ByteLevel"});
    for (chain, expected) in [
        (json!([replace, byte_level]), "a_b"),
        (json!([byte_level, replace]), "a b"),
        (
            json!([{"type": "Sequence", "decoders": [replace]}, byte_level]),
            "a_b",
        ),
    ] {
        let tokenizer = tokenizer(json!({"type": "Sequence", "decoders": chain}), false);
        assert_eq!(tokenizer.decode(&[97, 32, 98], false).unwrap(), expected);
    }
}

#[test]
fn byte_level_model_respects_other_decoders_and_no_decoder() {
    for (decoder, expected) in [
        (Value::Null, "a Ġ b"),
        (json!({"type": "Fuse"}), "aĠb"),
        (
            json!({"type": "Replace", "pattern": {"String": "Ġ"}, "content": "_"}),
            "a_b",
        ),
        (json!({"type": "Sequence", "decoders": []}), "aĠb"),
        (json!({"type": "ByteLevel"}), "a b"),
        (
            json!({"type": "Sequence", "decoders": [{"type": "ByteLevel"}]}),
            "a b",
        ),
    ] {
        let tokenizer = tokenizer(decoder, false);
        assert_eq!(tokenizer.decode(&[97, 32, 98], false).unwrap(), expected);
        assert_eq!(tokenizer.decode(&[], false).unwrap(), "");
    }
}

#[test]
fn byte_level_direct_and_sequence_keep_utf8_and_special_filtering() {
    for decoder in [
        json!({"type": "ByteLevel"}),
        json!({"type": "Sequence", "decoders": [{"type": "ByteLevel"}]}),
    ] {
        let tokenizer = tokenizer(decoder, false);
        for (ids, expected) in [
            (vec![195, 169], "é"),
            (vec![195], "�"),
            (vec![196, 128], "Ā"),
            (vec![u32::MAX], ""),
            (vec![], ""),
        ] {
            assert_eq!(tokenizer.decode(&ids, false).unwrap(), expected);
        }
        assert_eq!(tokenizer.decode(&[256, 97, 257], true).unwrap(), "a<tag>");
        assert_eq!(
            tokenizer.decode(&[256, 97, 257], false).unwrap(),
            "<s>a<tag>"
        );
    }
}
