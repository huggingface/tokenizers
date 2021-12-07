//! The general Rust <-> Java interop strategy is as follows:
//! - Data passed to function calls are passed by reference.
//! - Data returned from function calls is passed by reference.
//!
//! We used safer-ffi to build an unsafe-free C-compatible interface to the fast tokenizers.
//!
//! Typically, the data is owned on the Rust side and needs to be explicitly released in Rust.

extern crate tokenizers as tk;

use ::safer_ffi::prelude::*;
use tk::tokenizer::{EncodeInput, Encoding};
use tk::FromPretrainedParameters;
use tk::Tokenizer;

/// FFIResult is the FFI representation of Rust's Result type.
///
/// Note: value and error cannot both be None at the same time, nor be Some at the same time.
#[derive_ReprC]
#[repr(C)]
pub struct FFIResult<T> {
    /// value = None if the Result is not OK
    value: Option<repr_c::Box<T>>,
    /// error = None if the Result is OK
    error: Option<char_p::Box>,
}

/// FFITokenizer is the FFI representation of fast-tokenizer Tokenizer instance.
#[derive_ReprC]
#[ReprC::opaque]
pub struct FFITokenizer {
    tokenizer: Tokenizer,
}

/// Fast tokenizers supports various inputs
/// (e.g. single input, paired input, partially-pre-tokenized, batch)
///
/// This version currently only supports single input and batch
pub enum InputSequence<'s> {
    Str(&'s str),
    VecStr(&'s [&'s str]),
}

impl FFITokenizer {
    pub fn from_pretrained(identifier: &str) -> FFIResult<FFITokenizer> {
        let parameters = FromPretrainedParameters::default();
        let tk_result = Tokenizer::from_pretrained(identifier, Some(parameters));
        let error = (&tk_result)
            .as_ref()
            .err()
            .map(|e| char_p::new(e.to_string()));
        let value = tk_result
            .ok()
            .map(|tokenizer| repr_c::Box::new(FFITokenizer { tokenizer }));
        FFIResult { value, error }
    }

    pub fn encode(
        &self,
        input: &InputSequence,
        add_special_tokens: bool,
    ) -> FFIResult<FFIEncoding> {
        let input_sequence = match *input {
            InputSequence::Str(sequence) => tk::InputSequence::from(sequence),
            InputSequence::VecStr(sequence) => tk::InputSequence::from(sequence),
        };

        let single_input_sequence = EncodeInput::Single(input_sequence);
        let enc_result = self
            .tokenizer
            .encode(single_input_sequence, add_special_tokens);
        let error = (&enc_result)
            .as_ref()
            .err()
            .map(|e| char_p::new(e.to_string()));
        let value = enc_result
            .ok()
            .map(|e| FFIEncoding::from_rust(&e))
            .map(repr_c::Box::new);
        return FFIResult { value, error };
    }

    pub fn encode_batch(
        &self,
        input: &Vec<InputSequence>,
        add_special_tokens: bool,
    ) -> FFIResult<repr_c::Vec<repr_c::Box<FFIEncoding>>> {
        let encode_inputs: Vec<tk::EncodeInput> = input
            .iter()
            .map(|w| match *w {
                InputSequence::Str(s) => EncodeInput::Single(tk::InputSequence::from(s)),
                InputSequence::VecStr(s) => EncodeInput::Single(tk::InputSequence::from(s)),
            })
            .collect::<Vec<_>>();

        let enc_result = self
            .tokenizer
            .encode_batch(encode_inputs, add_special_tokens);
        let error = (&enc_result)
            .as_ref()
            .err()
            .map(|e| char_p::new(e.to_string()));
        let value = enc_result.ok().map(|encoded| {
            let ffi_encodings = encoded
                .iter()
                .map(|e| repr_c::Box::new(FFIEncoding::from_rust(e)))
                .collect::<Vec<_>>()
                .into();
            repr_c::Box::new(ffi_encodings)
        });

        return FFIResult { value, error };
    }
}

/// FFIEncoding is the FFI-representation of Encoding
///
/// Fast tokenizers typically uses u32 for ids. Java however does not support unsigned integers
/// as a primitive datatype natively. That's why those cases are converted to i64.
///
/// For the cases an Option<u32> is returned, we map the None case to -1.
#[derive_ReprC]
#[repr(C)]
pub struct FFIEncoding {
    ids: repr_c::Vec<i64>,
    type_ids: repr_c::Vec<i64>,
    tokens: repr_c::Vec<char_p::Box>,
    words: repr_c::Vec<i64>,
}

impl FFIEncoding {
    fn from_rust(enc: &Encoding) -> FFIEncoding {
        let m_ids = enc.get_ids().iter().map(|i| i64::from(*i));
        let ids = m_ids.collect::<Vec<_>>().into();
        let m_type_ids = enc.get_type_ids().iter().map(|i| i64::from(*i));
        let type_ids = m_type_ids.collect::<Vec<_>>().into();
        let m_tokens = enc.get_tokens().iter().map(|s| char_p::new(s.clone()));
        let tokens = m_tokens.collect::<Vec<_>>().into();
        let m_words = enc.get_word_ids().iter().map(|w| match w {
            Some(v) => i64::from(*v),
            None => -1, // to indicate null
        });
        let words = m_words.collect::<Vec<_>>().into();

        FFIEncoding {
            ids,
            type_ids,
            tokens,
            words,
        }
    }
}

#[ffi_export]
fn tokenizer_from_pretrained(ffi_identifier: char_p::Ref) -> repr_c::Box<FFIResult<FFITokenizer>> {
    let input = ffi_identifier.to_str();
    repr_c::Box::new(FFITokenizer::from_pretrained(input))
}

#[ffi_export]
fn encode_from_str(
    it: &FFITokenizer,
    ffi_input: char_p::Ref,
    add_special_tokens: bool,
) -> repr_c::Box<FFIResult<FFIEncoding>> {
    let str_input = ffi_input.to_str();
    let input = InputSequence::Str(str_input);
    let encoded = it.encode(&input, add_special_tokens);
    repr_c::Box::new(encoded)
}

#[ffi_export]
fn encode_batch(
    it: &FFITokenizer,
    ffi_input: &repr_c::Vec<char_p::Ref>,
    add_special_tokens: bool,
) -> repr_c::Box<FFIResult<repr_c::Vec<repr_c::Box<FFIEncoding>>>> {
    let input: Vec<InputSequence> = ffi_input
        .iter()
        .map(|w| InputSequence::Str(w.to_str()))
        .collect::<Vec<_>>();
    let encoded = it.encode_batch(&input, add_special_tokens);
    return repr_c::Box::new(encoded);
}

#[ffi_export]
fn tokenizer_drop(ptr: repr_c::Box<FFIResult<FFITokenizer>>) {
    drop(ptr);
}

#[ffi_export]
fn encoding_drop(ptr: repr_c::Box<FFIResult<FFIEncoding>>) {
    drop(ptr);
}

#[ffi_export]
fn encodings_drop(ptr: repr_c::Box<FFIResult<repr_c::Vec<repr_c::Box<FFIEncoding>>>>) {
    drop(ptr);
}

/// The following test function is necessary for the header generation.
/// Headers are only needed during development. It helps inspecting the
/// needed JNA interface.
#[::safer_ffi::cfg_headers]
#[test]
fn generate_headers() -> ::std::io::Result<()> {
    ::safer_ffi::headers::builder()
        .to_file("tokenizers.h")?
        .generate()
}
