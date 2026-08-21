mod common;

use common::*;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::decoders::byte_level::ByteLevelDecoder;
use tokenizers::models::ModelWrapper;
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordlevel;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::delimiter::CharDelimiterSplit;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{SplitDelimiterBehavior, Tokenizer, TokenizerImpl};

#[test]
fn bpe_serde() {
    let bpe = get_byte_level_bpe();
    let ser = serde_json::to_string(&bpe).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(bpe, de);
}

// `WordPiece`, `WordLevel` and `Unigram` are reached through `ModelWrapper` here because that is
// what a `tokenizer.json` names; each also carries its own serde, in the `serialization.rs` next to
// it in `tk-encode`.
#[test]
fn wordpiece_serde() {
    let wordpiece = ModelWrapper::from(get_bert_wordpiece());
    let ser = serde_json::to_string(&wordpiece).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(wordpiece, de);
}

#[test]
fn wordlevel_serde() {
    let wordlevel =
        ModelWrapper::from(wordlevel::from_file("data/gpt2-vocab.json", "<unk>".into()).unwrap());
    let ser = serde_json::to_string(&wordlevel).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(wordlevel, de);
}

#[test]
fn normalizers() {
    // The wrapper is the unit under test in both directions, exactly as for the decoders below:
    // being untagged, it serializes to precisely what the leaf serializes to. That one fieldless
    // normalizer refuses another one's JSON is asserted in `tk-convert`'s
    // `normalizers::tests::a_fieldless_normalizer_requires_its_tag`.

    // Test unit struct
    let nfc: NormalizerWrapper = NFC.into();
    let nfc_ser = serde_json::to_string(&nfc).unwrap();
    assert_eq!(nfc_ser, r#"{"type":"NFC"}"#);
    // wrapper can can deserialize from inner
    let nfc_wrapped: NormalizerWrapper = serde_json::from_str(&nfc_ser).unwrap();
    match &nfc_wrapped {
        NormalizerWrapper::NFC(_) => (),
        _ => panic!("NFC wrapped with incorrect variant"),
    }
    let ser_wrapped = serde_json::to_string(&nfc_wrapped).unwrap();
    assert_eq!(ser_wrapped, nfc_ser);

    // Test non-empty roundtrip
    let bert: NormalizerWrapper = BertNormalizer::default().into();
    let bert_ser = serde_json::to_string(&bert).unwrap();
    assert_eq!(
        bert_ser,
        r#"{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true}"#
    );
    // wrapper can deserialize from inner serialization
    let bert_wrapped: NormalizerWrapper = serde_json::from_str(&bert_ser).unwrap();
    match &bert_wrapped {
        NormalizerWrapper::BertNormalizer(_) => (),
        _ => panic!("BertNormalizer wrapped with incorrect variant"),
    }
    // wrapped serializes same way as inner
    let ser_wrapped = serde_json::to_string(&bert_wrapped).unwrap();
    assert_eq!(ser_wrapped, bert_ser);
}

#[test]
fn processors() {
    // The wrapper is the unit under test in both directions; being untagged, it serializes to
    // exactly what `BertProcessing` serializes to.
    let bert: PostProcessorWrapper =
        BertProcessing::new(("SEP".into(), 0), ("CLS".into(), 0)).into();
    let bert_ser = serde_json::to_string(&bert).unwrap();
    assert_eq!(
        bert_ser,
        r#"{"type":"BertProcessing","sep":["SEP",0],"cls":["CLS",0]}"#
    );
    let bert_wrapped: PostProcessorWrapper = serde_json::from_str(&bert_ser).unwrap();
    match &bert_wrapped {
        PostProcessorWrapper::Bert(_) => (),
        _ => panic!("Bert wrapped with incorrect variant"),
    }
    let ser_wrapped = serde_json::to_string(&bert_wrapped).unwrap();
    assert_eq!(ser_wrapped, bert_ser);
}

#[test]
fn pretoks() {
    // The wrapper is the unit under test in both directions, the same way `decoders` below tests
    // `ByteLevelDecoder`.
    //
    // The two "X shouldn't be deserializable from Y" checks this test used to make at the leaf are
    // made here at the real entry point: `PreTokenizerWrapper`'s legacy fallback is an *untagged*
    // enum that tries its variants in declaration order, and `BertPreTokenizer` is declared before
    // `Whitespace`. So a `{"type":"Whitespace"}` object arriving at the `Whitespace` variant is
    // exactly the statement that `BertPreTokenizer` refused it first -- which is only true because
    // every pre-tokenizer *requires* its `"type"` tag.

    // Test unit struct
    let bert: PreTokenizerWrapper = BertPreTokenizer.into();
    let bert_ser = serde_json::to_string(&bert).unwrap();
    assert_eq!(bert_ser, r#"{"type":"BertPreTokenizer"}"#);
    // wrapper can deserialize from inner, and lands on the Bert variant rather than any of the other
    // field-less ones
    let bert_wrapped: PreTokenizerWrapper = serde_json::from_str(&bert_ser).unwrap();
    match &bert_wrapped {
        PreTokenizerWrapper::BertPreTokenizer(_) => (),
        _ => panic!("Bert wrapped with incorrect variant"),
    }
    let ser_wrapped = serde_json::to_string(&bert_wrapped).unwrap();
    assert_eq!(ser_wrapped, bert_ser);

    // Test non-empty roundtrip
    let ch: PreTokenizerWrapper = CharDelimiterSplit::new(' ').into();
    let ch_ser = serde_json::to_string(&ch).unwrap();
    assert_eq!(ch_ser, r#"{"type":"CharDelimiterSplit","delimiter":" "}"#);
    // `EnumType` has no `CharDelimiterSplit` variant, so this one loads through the untagged
    // fallback -- and still has to land on `Delimiter`.
    let ch_wrapped: PreTokenizerWrapper = serde_json::from_str(&ch_ser).unwrap();
    match &ch_wrapped {
        PreTokenizerWrapper::Delimiter(_) => (),
        _ => panic!("CharDelimiterSplit wrapped with incorrect variant"),
    }
    // wrapped serializes same way as inner
    let ser_wrapped = serde_json::to_string(&ch_wrapped).unwrap();
    assert_eq!(ser_wrapped, ch_ser);

    let wsp: PreTokenizerWrapper = Whitespace {}.into();
    let wsp_ser = serde_json::to_string(&wsp).unwrap();
    assert_eq!(wsp_ser, r#"{"type":"Whitespace"}"#);
    let wsp_wrapped: PreTokenizerWrapper = serde_json::from_str(&wsp_ser).unwrap();
    match &wsp_wrapped {
        PreTokenizerWrapper::Whitespace(_) => (),
        _ => panic!("Whitespace wrapped with incorrect variant"),
    }

    let pattern: SplitPattern = "[SEP]".into();
    let pretok: PreTokenizerWrapper = Split::new(pattern, SplitDelimiterBehavior::Isolated, false)
        .unwrap()
        .into();
    let pretok_str = serde_json::to_string(&pretok).unwrap();
    assert_eq!(
        pretok_str,
        r#"{"type":"Split","pattern":{"String":"[SEP]"},"behavior":"Isolated","invert":false}"#
    );
    assert_eq!(
        serde_json::from_str::<PreTokenizerWrapper>(&pretok_str).unwrap(),
        pretok
    );

    let pattern = SplitPattern::Regex("[SEP]".to_string());
    let pretok: PreTokenizerWrapper = Split::new(pattern, SplitDelimiterBehavior::Isolated, false)
        .unwrap()
        .into();
    let pretok_str = serde_json::to_string(&pretok).unwrap();
    assert_eq!(
        pretok_str,
        r#"{"type":"Split","pattern":{"Regex":"[SEP]"},"behavior":"Isolated","invert":false}"#
    );
    assert_eq!(
        serde_json::from_str::<PreTokenizerWrapper>(&pretok_str).unwrap(),
        pretok
    );
}

#[test]
fn decoders() {
    // The wrapper is the unit under test in both directions; being untagged, it serializes to
    // exactly what `ByteLevelDecoder` serializes to.
    let byte_level: DecoderWrapper = ByteLevelDecoder::default().into();
    let byte_level_ser = serde_json::to_string(&byte_level).unwrap();
    assert_eq!(
        byte_level_ser,
        r#"{"type":"ByteLevel"}"#
    );
    let byte_level_wrapper: DecoderWrapper = serde_json::from_str(&byte_level_ser).unwrap();
    match &byte_level_wrapper {
        DecoderWrapper::ByteLevel(_) => (),
        _ => panic!("ByteLevel wrapped with incorrect variant"),
    }
    let ser_wrapped = serde_json::to_string(&byte_level_wrapper).unwrap();
    assert_eq!(ser_wrapped, byte_level_ser);
}

#[test]
fn models() {
    let bpe = BPE::default();
    let bpe_ser = serde_json::to_string(&bpe).unwrap();
    serde_json::from_str::<BPE>(&bpe_ser).unwrap();
    let bpe_wrapper: ModelWrapper = serde_json::from_str(&bpe_ser).unwrap();
    match &bpe_wrapper {
        ModelWrapper::BPE(_) => (),
        _ => panic!("BPE wrapped with incorrect variant"),
    }
    let ser_wrapped = serde_json::to_string(&bpe_wrapper).unwrap();
    assert_eq!(ser_wrapped, bpe_ser);
}

#[test]
fn tokenizer() {
    // The model parameter is incidental here -- what is under test is the *normalizer* slot,
    // unwrapped (NFC) versus wrapped (NormalizerWrapper). `ModelWrapper` is what a `tokenizer.json`
    // names in the model slot, so it is what the round trip goes through.
    let wordpiece = WordPiece::default();
    let mut tokenizer = Tokenizer::new(wordpiece);
    tokenizer.with_normalizer(Some(NFC)).unwrap();
    let ser = serde_json::to_string(&tokenizer).unwrap();
    let _: Tokenizer = serde_json::from_str(&ser).unwrap();
    // That `NFKC` refuses `NFC`'s JSON is asserted in `tk-convert`'s `normalizers::tests`, where the
    // required-tag property it depends on is spelled out.
    let de: TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = serde_json::from_str(&ser).unwrap();
    assert_eq!(serde_json::to_string(&de).unwrap(), ser);
}

#[test]
fn bpe_with_dropout_serde() {
    let mut bpe = BPE {
        dropout: Some(0.1),
        ..Default::default()
    };
    let ser = serde_json::to_string(&bpe).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(bpe, de);

    // set dropout to 0.0 (which is analogous to None) and reserialize
    bpe.dropout = Some(0.0);
    let ser = serde_json::to_string(&bpe).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(bpe, de);
}

#[test]
fn test_deserialize_long_file() {
    let _tokenizer = Tokenizer::from_file("data/albert-base-v1-tokenizer.json").unwrap();
}
