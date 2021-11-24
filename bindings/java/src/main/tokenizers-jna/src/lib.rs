extern crate libc;
extern crate tokenizers as tk;

use std::borrow::Borrow;
use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;
use std::ptr::null_mut;

use tk::tokenizer::EncodeInput;
use tk::tokenizer::InputSequence;
use tk::{Encoding, Tokenizer};

use libc::boolean_t;
use std::ops::Deref;
use tk::FromPretrainedParameters;

//JInputSequence
//JEncoding

//from Vec<String>
//from String
//- When ``is_pretokenized=False``: :data:`~TextInputSequence` (InputSequence) union types
//struct TextInputSequence<'s>(tk::InputSequence<'s>);
type Result<T> = std::result::Result<T, JError>;
pub struct JError;

pub struct JInputSequence<'s> {
    pub input_sequence: tk::InputSequence<'s>,
}
//todo: make from pair
impl JInputSequence<'_> {
    pub fn from_str(st: &str) -> JInputSequence {
        let inputSequence = InputSequence::from(st);
        return JInputSequence {
            input_sequence: inputSequence,
        }
    }

    pub fn from_vec_str(vec: Vec<&str>) -> JInputSequence {
        let inputSequence = InputSequence::from(vec);
        return  JInputSequence {
            input_sequence: inputSequence,
        }
    }
}

pub struct JPairInputSequence<'s> {
    pub first: tk::InputSequence<'s>,
    pub second: tk::InputSequence<'s>,
}

pub struct JEncoding {
    pub encoding: Option<tk::tokenizer::Encoding>
}

impl JEncoding {

    pub fn get_ids(&self) -> &[u32] {
       let e = &self.encoding.as_ref().expect("Unitialized encoding");
        return e.get_ids();

    }

    pub fn get_tokens(&self) -> &[String] {
        let e = &self.encoding.as_ref().expect("Unitialized encoding");
        return e.get_tokens();
    }

}

pub struct JTokenizer {
    tokenizer: Option<Tokenizer>,
}

impl JTokenizer {
    //FromPretrainedParameters: two Option of Strings
    pub fn from_pretrained(identifier: &str) -> JTokenizer {
        let parameters = FromPretrainedParameters::default();
        let tokenizer = Tokenizer::from_pretrained(identifier, Some(parameters));
        match tokenizer {
            Ok(value) => {
                return JTokenizer {
                    tokenizer: Some(value),
                };
            }
            Err(error) => {
                println!("Problem instantiating tokenizer {:?}", error);
                return JTokenizer { tokenizer: None };
            }
        }
    }

    pub fn encode(&self, input: &JInputSequence) -> JEncoding {
        let singles = EncodeInput::Single(input.input_sequence.clone());
        match &self.tokenizer {
            Some(value) => {
                let encodings = value.encode(singles, true).ok();
                return JEncoding{ encoding: encodings};
            }
            None => {
                println!("cannot encode");
                return JEncoding{ encoding: None};
            }
        }
    }

    pub fn encode_pair(&self, pair: &JPairInputSequence) -> JEncoding {
        let first = pair.first.clone();
        let second = pair.second.clone();
        let pair = EncodeInput::Dual(first, second);
        match &self.tokenizer {
            Some(value) => {
                let encodings = value.encode(pair, true).ok();
                return JEncoding{ encoding: encodings};
            }
            None => {
                println!("cannot encode pair");
                return JEncoding{ encoding: None } ;
            }
        }
    }

    pub fn print_tokenizer(&self) {
        // let instance = unsafe{ &*(self.tokenizer) };
        match &self.tokenizer {
            Some(value) => {
                let string = value.to_string(true);
                println!("I was called in rust. tokenizer: {:?}", value);
            }
            None => {
                println!("no tokenizer found");
            }
        }
    }
}

// pub fn tokenize(&self) -> CEncodings {
//     let input = EncodeInput::Single(InputSequence::Raw(Cow::from("Hellow")));
//     let encodings = self.tokenizer.encode(input, false).unwrap();
//     let ids = encodings.get_ids();
//     CEncodings{ ids: ids.to_vec().clone()}
//     //println!("doing a thing! also, number is {}!", self.number);
// }

//maybe inject handle pointer instead (PointerByReference in jna)
// and return error code if allocation fails
//assert pointer not null
#[no_mangle]
pub unsafe extern "C" fn JTokenizer_from_pretrained(identifier: *mut c_char) -> *mut JTokenizer {
    let cstr = unsafe { CString::from_raw(identifier).to_string_lossy().to_string() };
    let boxed_a = Box::new(JTokenizer::from_pretrained(&cstr));
    Box::into_raw(boxed_a)
}

#[no_mangle]
pub unsafe extern "C" fn JTokenizer_drop(p: *mut JTokenizer) {
    Box::from_raw(p);
}

#[no_mangle]
pub unsafe extern "C" fn JTokenizer_print_tokenizer(a: *mut JTokenizer) {
    let a = &*a;
    a.print_tokenizer();
}

// pub fn print_string(my_string: &str) {
//     println!("ccz {:?}", my_string);
// }
//
// #[no_mangle]
// pub extern "C" fn rust_function(value: *mut c_char) {
//     let cstr = unsafe { CString::from_raw(value).to_string_lossy().to_string() };
//     let reference = &cstr;
//     print_string(reference);
// }
