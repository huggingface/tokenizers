extern crate tokenizers as tk;

use ::safer_ffi::prelude::*;

use tk::Tokenizer;
use tk::FromPretrainedParameters;

#[derive_ReprC]
#[ReprC::opaque]
pub struct FFITokenizer {
    tokenizer: Tokenizer
}

#[ffi_export]
fn tokenizer_new() -> repr_c::Box<FFITokenizer>
{
    let identifier = "setu4993/LaBSE";
    let parameters = FromPretrainedParameters::default();
    let tk_result = Tokenizer::from_pretrained(identifier, Some(parameters));
    match tk_result {
        Ok(v) => repr_c::Box::new(FFITokenizer { tokenizer: v }),
        Err(_e) => panic!("identifier not found"),
    }
}

#[ffi_export]
fn tokenizer_drop(ptr: repr_c::Box<FFITokenizer>)
{
    drop(ptr);
}

#[derive_ReprC]
#[repr(C)]
pub struct FFIEncoding {
    ids: repr_c::Vec<u32>,
    type_ids: repr_c::Vec<u32>,
    foo: repr_c::Vec<Option<repr_c::Box<u32>>>,
    tokens: repr_c::Vec<repr_c::String>,
    words: repr_c::Vec<Option<repr_c::Box<u32>>>,
    special_token_mask: repr_c::Vec<u32>,
    attention_mask: repr_c::Vec<u32>
}

#[ffi_export]
fn encoding_drop(ptr: repr_c::Box<FFIEncoding>) 
{
    drop(ptr);
}

/// The following test function is necessary for the header generation. 
/// Headers are only needed during development. It helps inspecting the 
/// needed JNA interface.
#[::safer_ffi::cfg_headers]
#[test]
fn generate_headers () -> ::std::io::Result<()>
{
    ::safer_ffi::headers::builder()
        .to_file("tokenizers.h")?
        .generate()
}
