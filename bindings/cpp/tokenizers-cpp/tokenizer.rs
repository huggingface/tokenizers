#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    pub enum TruncationStrategy {
        LongestFirst,
        OnlyFirst,
        OnlySecond,
    }

    pub enum PaddingDirection {
        Left,
        Right,
    }

    pub enum OffsetType {
        Byte,
        Char,
    }

    pub enum InputSequenceTag {
        Str,
        String,
        StringVec,
        StringSlice,
    }

    unsafe extern "C++" {
        include!("tokenizers-cpp/input_sequence.h");
        // Can probably be declared as shared when enums with data are supported
        type InputSequence;
        type InputSequenceTag;
        fn get_tag(self: &InputSequence) -> InputSequenceTag;
        // these should ideally be `unsafe`, but rustfmt removes the keyword
        fn get_str(self: &InputSequence) -> &str;
        fn get_string(self: &InputSequence) -> String;
        fn get_string_vec(self: &InputSequence) -> Vec<String>;
        fn get_string_slice(self: &InputSequence) -> &[String];

        type InputSequencePair;
        fn first(self: &InputSequencePair) -> &InputSequence;
        fn second(self: &InputSequencePair) -> &InputSequence;
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    unsafe extern "C++" {
        include!("tokenizers-cpp/tokenizer.h");
        type Normalizer = crate::normalizers::Normalizer;
        type PreTokenizer = crate::pre_tokenizers::PreTokenizer;
        type Model = crate::models::Model;
        type PostProcessor = crate::processors::PostProcessor;
        type Decoder = crate::decoders::Decoder;
        type OptionU32 = crate::models::ffi::OptionU32;
        type OptionString = crate::models::ffi::OptionString;
        #[namespace = "huggingface::tokenizers"]
        type TokenAndId = crate::models::ffi::TokenAndId;
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Encoding_1;
        type Tokenizer;

        // FIXME many of the below functions should take Box, not &.
        //  Look for clone() in the implementations.
        fn tokenizer(model: &Model) -> Box<Tokenizer>;
        fn set_normalizer(tokenizer: &mut Tokenizer, normalizer: &Normalizer);
        fn set_pre_tokenizer(tokenizer: &mut Tokenizer, pre_tokenizer: &PreTokenizer);
        fn set_post_processor(tokenizer: &mut Tokenizer, post_processor: &PostProcessor);
        fn set_decoder(tokenizer: &mut Tokenizer, decoder: &Decoder);
        fn set_padding(
            tokenizer: &mut Tokenizer,
            is_fixed_length: bool,
            fixed_length: usize,
            direction: PaddingDirection,
            pad_to_multiple_of: usize,
            pad_id: u32,
            pad_type_id: u32,
            pad_token: String,
        );
        fn set_no_padding(tokenizer: &mut Tokenizer);
        fn set_truncation(
            tokenizer: &mut Tokenizer,
            max_length: usize,
            strategy: TruncationStrategy,
            stride: usize,
        );
        fn set_no_truncation(tokenizer: &mut Tokenizer);

        fn token_to_id(tokenizer: &Tokenizer, token: &str) -> OptionU32;
        fn id_to_token(tokenizer: &Tokenizer, id: u32) -> OptionString;
        fn get_vocab(tokenizer: &Tokenizer, with_added_tokens: bool) -> Vec<TokenAndId>;
        fn get_vocab_size(tokenizer: &Tokenizer, with_added_tokens: bool) -> usize;

        fn encode(
            tokenizer: &Tokenizer,
            input: &InputSequence,
            add_special_tokens: bool,
            offset_type: OffsetType,
        ) -> Result<Box<Encoding_1>>;

        fn encode_pair(
            tokenizer: &Tokenizer,
            input: &InputSequencePair,
            add_special_tokens: bool,
            offset_type: OffsetType,
        ) -> Result<Box<Encoding_1>>;

        // TODO use &[InputSequence] when it's supported (similar for encode_pair_batch)
        fn encode_batch(
            tokenizer: &Tokenizer,
            input: &CxxVector<InputSequence>,
            add_special_tokens: bool,
            offset_type: OffsetType,
        ) -> Result<Vec<Encoding_1>>;

        fn encode_pair_batch(
            tokenizer: &Tokenizer,
            input: &CxxVector<InputSequencePair>,
            add_special_tokens: bool,
            offset_type: OffsetType,
        ) -> Result<Vec<Encoding_1>>;

        fn decode(
            tokenizer: &Tokenizer,
            ids: Vec<u32>,
            skip_special_tokens: bool,
        ) -> Result<String>;

        fn decode_batch(
            tokenizer: &Tokenizer,
            ids: Vec<u32>,
            sequence_starts: &[usize],
            skip_special_tokens: bool,
        ) -> Result<Vec<String>>;
    }
}

use crate::{forward_cxx_enum, impl_extern_type, models::vocab_to_vec, wrap_option};
use cxx::CxxVector;
use derive_more::{Deref, DerefMut};
use ffi::*;
use tk::{EncodeInput, PaddingParams, PaddingStrategy, Result, TruncationParams};

#[derive(Deref, DerefMut)]
#[allow(non_camel_case_types)]
struct Encoding_1(tk::Encoding);

impl_extern_type!(Normalizer, "huggingface::tokenizers::ffi::Normalizer");

impl_extern_type!(Model, "huggingface::tokenizers::ffi::Model");

impl_extern_type!(PreTokenizer, "huggingface::tokenizers::ffi::PreTokenizer");

impl_extern_type!(PostProcessor, "huggingface::tokenizers::ffi::PostProcessor");

impl_extern_type!(Decoder, "huggingface::tokenizers::ffi::Decoder");

#[derive(Deref, DerefMut)]
struct Tokenizer(tk::TokenizerImpl<Model, Normalizer, PreTokenizer, PostProcessor, Decoder>);

fn tokenizer(model: &Model) -> Box<Tokenizer> {
    Box::new(Tokenizer(tk::TokenizerImpl::new(model.clone())))
}

fn set_normalizer(tokenizer: &mut Tokenizer, normalizer: &Normalizer) {
    tokenizer.with_normalizer(normalizer.clone());
}

fn set_pre_tokenizer(tokenizer: &mut Tokenizer, pre_tokenizer: &PreTokenizer) {
    tokenizer.with_pre_tokenizer(pre_tokenizer.clone());
}

fn set_post_processor(tokenizer: &mut Tokenizer, post_processor: &PostProcessor) {
    tokenizer.with_post_processor(post_processor.clone());
}

fn set_decoder(tokenizer: &mut Tokenizer, decoder: &Decoder) {
    tokenizer.with_decoder(decoder.clone());
}

fn set_padding(
    tokenizer: &mut Tokenizer,
    is_fixed_length: bool,
    fixed_length: usize,
    direction: PaddingDirection,
    pad_to_multiple_of: usize,
    pad_id: u32,
    pad_type_id: u32,
    pad_token: String,
) {
    tokenizer.with_padding(Some(PaddingParams {
        strategy: if is_fixed_length {
            PaddingStrategy::Fixed(fixed_length)
        } else {
            PaddingStrategy::BatchLongest
        },
        direction: forward_cxx_enum!(direction, PaddingDirection, Left, Right),
        // note tk::PaddingParams considers None and Some(0) the same
        pad_to_multiple_of: Some(pad_to_multiple_of),
        pad_id,
        pad_type_id,
        pad_token,
    }));
}

fn set_no_padding(tokenizer: &mut Tokenizer) {
    tokenizer.with_padding(None);
}

fn set_truncation(
    tokenizer: &mut Tokenizer,
    max_length: usize,
    strategy: TruncationStrategy,
    stride: usize,
) {
    tokenizer.with_truncation(Some(TruncationParams {
        max_length,
        strategy: forward_cxx_enum!(
            strategy,
            TruncationStrategy,
            LongestFirst,
            OnlyFirst,
            OnlySecond
        ),
        stride,
    }));
}

fn set_no_truncation(tokenizer: &mut Tokenizer) {
    tokenizer.with_truncation(None);
}

fn id_to_token(tokenizer: &Tokenizer, id: u32) -> OptionString {
    wrap_option!(tokenizer.id_to_token(id), OptionString, "".to_string())
}

fn token_to_id(tokenizer: &Tokenizer, token: &str) -> OptionU32 {
    wrap_option!(tokenizer.token_to_id(token), OptionU32, 0)
}

fn get_vocab(tokenizer: &Tokenizer, with_added_tokens: bool) -> Vec<TokenAndId> {
    vocab_to_vec(tokenizer.get_vocab(with_added_tokens))
}

fn get_vocab_size(tokenizer: &Tokenizer, with_added_tokens: bool) -> usize {
    tokenizer.get_vocab_size(with_added_tokens)
}

impl<'s> From<&'s InputSequence> for tk::InputSequence<'s> {
    fn from(input: &'s InputSequence) -> Self {
        match input.get_tag() {
            InputSequenceTag::Str => input.get_str().into(),
            InputSequenceTag::String => input.get_string().into(),
            InputSequenceTag::StringSlice => input.get_string_slice().into(),
            InputSequenceTag::StringVec => input.get_string_vec().into(),
            x => panic!("Illegal InputSequenceTag value {}", x.repr),
        }
    }
}

impl<'s> From<&'s InputSequencePair> for EncodeInput<'s> {
    fn from(input: &'s InputSequencePair) -> Self {
        (input.first(), input.second()).into()
    }
}

fn encode_impl<'s, I: Into<EncodeInput<'s>>>(
    tokenizer: &Tokenizer,
    input: I,
    add_special_tokens: bool,
    offset_type: OffsetType,
) -> Result<Box<Encoding_1>> {
    Ok(Box::new(Encoding_1(match offset_type {
        OffsetType::Byte => tokenizer.encode(input, add_special_tokens)?,
        OffsetType::Char => tokenizer.encode_char_offsets(input, add_special_tokens)?,
        x => panic!("Illegal OffsetType value {}", x.repr),
    })))
}

fn encode(
    tokenizer: &Tokenizer,
    input: &InputSequence,
    add_special_tokens: bool,
    offset_type: OffsetType,
) -> Result<Box<Encoding_1>> {
    encode_impl(tokenizer, input, add_special_tokens, offset_type)
}

fn encode_pair(
    tokenizer: &Tokenizer,
    input: &InputSequencePair,
    add_special_tokens: bool,
    offset_type: OffsetType,
) -> Result<Box<Encoding_1>> {
    encode_impl(tokenizer, input, add_special_tokens, offset_type)
}

// TODO https://github.com/dtolnay/cxx/issues/547
// fn encode_batch_impl<'s, I: Into<EncodeInput<'s>> + VectorElement>(
//     tokenizer: &Tokenizer,
//     input: &CxxVector<I>,
//     add_special_tokens: bool,
//     offset_type: OffsetType,
// ) -> Result<Vec<Encoding_1>> {
//     let input: Vec<_> = input.iter().map(|x| EncodeInput::from(x)).collect();
//     Ok((match offset_type {
//         OffsetType::Byte => tokenizer.encode_batch(input, add_special_tokens),
//         OffsetType::Char => tokenizer.encode_batch_char_offsets(input, add_special_tokens),
//         x => panic!("Illegal OffsetType value {}", x.repr),
//     })?
//     .into_iter()
//     .map(|x| Encoding_1(x))
//     .collect())
// }

fn encode_batch(
    tokenizer: &Tokenizer,
    input: &CxxVector<InputSequence>,
    add_special_tokens: bool,
    offset_type: OffsetType,
) -> Result<Vec<Encoding_1>> {
    let input: Vec<_> = input.iter().map(|x| EncodeInput::from(x)).collect();
    Ok((match offset_type {
        OffsetType::Byte => tokenizer.encode_batch(input, add_special_tokens),
        OffsetType::Char => tokenizer.encode_batch_char_offsets(input, add_special_tokens),
        x => panic!("Illegal OffsetType value {}", x.repr),
    })?
    .into_iter()
    .map(|x| Encoding_1(x))
    .collect())
}

fn encode_pair_batch(
    tokenizer: &Tokenizer,
    input: &CxxVector<InputSequencePair>,
    add_special_tokens: bool,
    offset_type: OffsetType,
) -> Result<Vec<Encoding_1>> {
    let input: Vec<_> = input.iter().map(|x| EncodeInput::from(x)).collect();
    Ok((match offset_type {
        OffsetType::Byte => tokenizer.encode_batch(input, add_special_tokens),
        OffsetType::Char => tokenizer.encode_batch_char_offsets(input, add_special_tokens),
        x => panic!("Illegal OffsetType value {}", x.repr),
    })?
    .into_iter()
    .map(|x| Encoding_1(x))
    .collect())
}

fn decode(tokenizer: &Tokenizer, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String> {
    tokenizer.decode(ids, skip_special_tokens)
}

fn decode_batch(
    tokenizer: &Tokenizer,
    mut ids: Vec<u32>,
    sequence_lengths: &[usize],
    skip_special_tokens: bool,
) -> Result<Vec<String>> {
    let total_len = ids.len();
    let sequences: Vec<Vec<u32>> = sequence_lengths
        .iter()
        .rev()
        .map(|len| ids.split_off(total_len - len))
        .collect();
    tokenizer.decode_batch(sequences, skip_special_tokens)
}
