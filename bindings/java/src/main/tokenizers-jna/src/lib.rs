extern crate libc;

use std::ffi::CStr;
use std::os::raw::c_char;
use tokenizers as tk;

// use std::fs::File;
// use std::io::{BufRead, BufReader};
// use std::path::Path;

use tokenizers::models::wordpiece::{WordPiece, WordPieceTrainerBuilder};
use tokenizers::normalizers::{BertNormalizer, NormalizerWrapper};
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{decoders, EncodeInput, Encoding, InputSequence, Model, TokenizerImpl};

use tokenizers::decoders::DecoderWrapper;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::PostProcessorWrapper;
use std::borrow::Cow;

//Simulating a struct with rust types not very ffi friendly
struct FakeBertTokenizer {
    weird_rust_variable: Vec<String>
}

impl FakeBertTokenizer {

    fn new() -> FakeBertTokenizer {
        let mut vec = Vec::new();
        vec.push("hi".to_string());
        vec.push("again".to_string());
        FakeBertTokenizer {weird_rust_variable: vec}
    }
}

// type BertTokenizer = TokenizerImpl<
//     WordPiece,
//     BertNormalizer,
//     BertPreTokenizer,
//     BertProcessing,
//     decoders::wordpiece::WordPiece,
// >;
//
// //create builder pattern for the config?

//
// pub struct CBertTokenizer {
//     pub tokenizer: *mut BertTokenizer,
// }

//Opacque to the rust structs
pub struct CBertTokenizer {
    tokenizer: *mut FakeBertTokenizer,
}

// Return types
// pub struct CEncodings {
//     ids: *mut i32,
//     size_ids: i32
// }

impl CBertTokenizer {

    //vocabulary_path: &str
    // pub fn new() -> CBertTokenizer {
    //     let wp = WordPiece::from_file("data/path/blabla")
    //         .build()
    //         .unwrap();
    //     let tokenizer = create_bert_tokenizer(wp);
    //     let boxed_tokenizer = Box::new(tokenizer);
    //     let raw = Box::into_raw(boxed_tokenizer);
    //    return  CBertTokenizer{tokenizer: raw}
    // }

    pub fn new() -> CBertTokenizer {
        let fake_tokenizer = FakeBertTokenizer::new();
        let boxed_tokenizer = Box::new(fake_tokenizer);
        let raw = Box::into_raw(boxed_tokenizer);
        return CBertTokenizer{tokenizer: raw}
    }

    pub fn some_method(&self)  {
        let number = 23;
        let instance = unsafe{ &*(self.tokenizer) };
        println!("I was called in rust. vector: {:?}", instance.weird_rust_variable);
    }

    // pub fn tokenize(&self) -> CEncodings {
    //     let input = EncodeInput::Single(InputSequence::Raw(Cow::from("Hellow")));
    //     let encodings = self.tokenizer.encode(input, false).unwrap();
    //     let ids = encodings.get_ids();
    //     CEncodings{ ids: ids.to_vec().clone()}
    //     //println!("doing a thing! also, number is {}!", self.number);
    // }

}
/// from hf repository: Resembling the BertTokenizer implementation from the Python bindings.
// fn create_bert_tokenizer(wp: WordPiece) -> BertTokenizer {
//     let sep_id = *wp.get_vocab().get("[SEP]").unwrap();
//     let cls_id = *wp.get_vocab().get("[CLS]").unwrap();
//     let mut tokenizer = TokenizerImpl::new(wp);
//     tokenizer.with_pre_tokenizer(BertPreTokenizer);
//     tokenizer.with_normalizer(BertNormalizer::default());
//     tokenizer.with_decoder(decoders::wordpiece::WordPiece::default());
//     tokenizer.with_post_processor(BertProcessing::new(
//         ("[SEP]".to_string(), sep_id),
//         ("[CLS]".to_string(), cls_id),
//     ));
//     tokenizer
// }

//instead of the opacque pointers, we can use this instead
// static mut C_BERT_TOKENIZER: CBertTokenizer = CBertTokenizer {
//     serial: Some(SerialPort),
// };


//maybe inject handle pointer instead (PointerByReference in jna)
// and return error code if allocation fails
#[no_mangle]
pub extern  "C" fn CBertTokenizer_new() -> *mut CBertTokenizer {
    let boxed_a = Box::new(CBertTokenizer::new() );
    Box::into_raw(boxed_a)
}

#[no_mangle]
pub unsafe extern "C" fn CBertTokenizer_drop(a: *mut CBertTokenizer) {
    Box::from_raw(a);
}

#[no_mangle]
pub unsafe extern "C" fn CBertTokenizer_some_method(a: *mut CBertTokenizer) {
    let a = &*a;
    a.some_method()
}

#[no_mangle]
pub extern "C" fn rust_function(value: *const c_char) {
    let cstr = unsafe { CStr::from_ptr(value) };
    match cstr.to_str() {
        Ok(value) => {
            println!("my string: {}", value);
        }
        Err(error) => {
            panic!("Problem converting string {:?}", error)
        }
    }

}