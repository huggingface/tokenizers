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
fn create_tokenizer() -> repr_c::Box<FFITokenizer>
{
    let identifier = "setu4993/LaBSE";
    let parameters = FromPretrainedParameters::default();
    let tk_result = Tokenizer::from_pretrained(identifier, Some(parameters));
    match tk_result {
        Ok(v) => repr_c::Box::new(FFITokenizer { tokenizer: v }),
        Err(e) => panic!("identifier not found"),
    }
}

#[ffi_export]
fn destroy_tokenizer(ptr: repr_c::Box<FFITokenizer>)
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
