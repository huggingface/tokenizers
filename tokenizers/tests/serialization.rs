mod common;

use common::*;
use tokenizers::decoders::byte_level::ByteLevel;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::normalizers::unicode::{NFC, NFKC};
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::delimiter::CharDelimiterSplit;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{Tokenizer, TokenizerImpl};

#[test]
fn bpe_serde() {
    let bpe = get_byte_level_bpe();
    let ser = serde_json::to_string(&bpe).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(bpe, de);
}

#[test]
fn wordpiece_serde() {
    let wordpiece = get_bert_wordpiece();
    let ser = serde_json::to_string(&wordpiece).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(wordpiece, de);
}

#[test]
fn wordlevel_serde() {
    let wordlevel = WordLevel::from_file("data/gpt2-vocab.json", "<unk>".into()).unwrap();
    let ser = serde_json::to_string(&wordlevel).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(wordlevel, de);
}

#[test]
fn normalizers() {
    // Test unit struct
    let nfc = NFC;
    let nfc_ser = serde_json::to_string(&nfc).unwrap();
    // empty struct can deserialize from self
    serde_json::from_str::<NFC>(&nfc_ser).unwrap();
    let err: Result<NFKC, _> = serde_json::from_str(&nfc_ser);
    assert!(err.is_err(), "NFKC shouldn't be deserializable from NFC");
    // wrapper can can deserialize from inner
    let nfc_wrapped: NormalizerWrapper = serde_json::from_str(&nfc_ser).unwrap();
    match &nfc_wrapped {
        NormalizerWrapper::NFC(_) => (),
        _ => panic!("NFC wrapped with incorrect variant"),
    }
    let ser_wrapped = serde_json::to_string(&nfc_wrapped).unwrap();
    assert_eq!(ser_wrapped, nfc_ser);

    // Test non-empty roundtrip
    let bert = BertNormalizer::default();
    let bert_ser = serde_json::to_string(&bert).unwrap();
    // make sure we can deserialize to self
    serde_json::from_str::<BertNormalizer>(&bert_ser).unwrap();
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
    let bert = BertProcessing::new(("SEP".into(), 0), ("CLS".into(), 0));
    let bert_ser = serde_json::to_string(&bert).unwrap();
    serde_json::from_str::<BertProcessing>(&bert_ser).unwrap();
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
    // Test unit struct
    let bert = BertPreTokenizer;
    let bert_ser = serde_json::to_string(&bert).unwrap();
    // empty struct can deserialize from self
    serde_json::from_str::<BertPreTokenizer>(&bert_ser).unwrap();
    let err: Result<Whitespace, _> = serde_json::from_str(&bert_ser);
    assert!(
        err.is_err(),
        "Whitespace shouldn't be deserializable from BertPreTokenizer"
    );
    // wrapper can can deserialize from inner
    let bert_wrapped: PreTokenizerWrapper = serde_json::from_str(&bert_ser).unwrap();
    match &bert_wrapped {
        PreTokenizerWrapper::BertPreTokenizer(_) => (),
        _ => panic!("Bert wrapped with incorrect variant"),
    }
    let ser_wrapped = serde_json::to_string(&bert_wrapped).unwrap();
    assert_eq!(ser_wrapped, bert_ser);

    // Test non-empty roundtrip
    let ch = CharDelimiterSplit::new(' ');
    let ch_ser = serde_json::to_string(&ch).unwrap();
    // make sure we can deserialize to self
    serde_json::from_str::<CharDelimiterSplit>(&ch_ser).unwrap();
    // wrapper can deserialize from inner serialization
    let ch_wrapped: PreTokenizerWrapper = serde_json::from_str(&ch_ser).unwrap();
    match &ch_wrapped {
        PreTokenizerWrapper::Delimiter(_) => (),
        _ => panic!("CharDelimiterSplit wrapped with incorrect variant"),
    }
    // wrapped serializes same way as inner
    let ser_wrapped = serde_json::to_string(&ch_wrapped).unwrap();
    assert_eq!(ser_wrapped, ch_ser);

    let wsp = Whitespace::default();
    let wsp_ser = serde_json::to_string(&wsp).unwrap();
    serde_json::from_str::<Whitespace>(&wsp_ser).unwrap();
    let err: Result<BertPreTokenizer, _> = serde_json::from_str(&wsp_ser);
    assert!(
        err.is_err(),
        "BertPreTokenizer shouldn't be deserializable from Whitespace"
    );
}

#[test]
fn decoders() {
    let byte_level = ByteLevel::default();
    let byte_level_ser = serde_json::to_string(&byte_level).unwrap();
    serde_json::from_str::<ByteLevel>(&byte_level_ser).unwrap();
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
    let wordpiece = WordPiece::default();
    let mut tokenizer = Tokenizer::new(wordpiece);
    tokenizer.with_normalizer(NFC);
    let ser = serde_json::to_string(&tokenizer).unwrap();
    let _: Tokenizer = serde_json::from_str(&ser).unwrap();
    let unwrapped_nfc_tok: TokenizerImpl<
        WordPiece,
        NFC,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = serde_json::from_str(&ser).unwrap();
    assert_eq!(serde_json::to_string(&unwrapped_nfc_tok).unwrap(), ser);
    let err: Result<
        TokenizerImpl<WordPiece, NFKC, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>,
        _,
    > = serde_json::from_str(&ser);
    assert!(err.is_err(), "NFKC shouldn't be deserializable from NFC");
    let de: TokenizerImpl<
        WordPiece,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = serde_json::from_str(&ser).unwrap();
    assert_eq!(serde_json::to_string(&de).unwrap(), ser);
}

#[test]
fn test_deserialize_long_file() {
    let _tokenizer = Tokenizer::from_file("data/albert-base-v1-tokenizer.json").unwrap();
}
