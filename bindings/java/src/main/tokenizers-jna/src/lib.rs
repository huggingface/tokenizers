extern crate libc;
extern crate tokenizers as tk;

use std::borrow::Borrow;
use std::ffi::CString;
use std::ffi::CStr;
use std::{fmt, u32};
use std::os::raw::c_char;
use std::ptr::null_mut;

use tk::tokenizer::EncodeInput;
use tk::tokenizer::InputSequence;
use tk::{Encoding, Tokenizer};

use libc::{boolean_t, size_t};
use std::ops::Deref;
use tk::FromPretrainedParameters;
use std::mem::ManuallyDrop;
use std::slice;

//JInputSequence
//JEncoding

//from Vec<String>
//from String
//- When ``is_pretokenized=False``: :data:`~TextInputSequence` (InputSequence) union types
//struct TextInputSequence<'s>(tk::InputSequence<'s>);
type Result<T> = std::result::Result<T, JError>;
pub struct JError;

//remove the J as its private and the pub
pub struct JInputSequence<'s> {
    pub input_sequence: tk::InputSequence<'s>,
}
//todo: make from pair
impl JInputSequence<'_> {
    pub fn from_str(st: String) -> JInputSequence<'static> {
        let inputSequence = InputSequence::from(st);
        return JInputSequence {
            input_sequence: inputSequence,
        }
    }

    pub fn from_vec_str(vec: Vec<String>) -> JInputSequence<'static> {
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

    //get length

    pub fn get_length(&self) -> usize {
        let e = &self.encoding.as_ref().expect("Unitialized encoding");
        return e.get_ids().to_vec().len();
    }

    pub fn get_ids(&self) -> Vec<u32> {
       let e = &self.encoding.as_ref().expect("Unitialized encoding");
        return e.get_ids().to_vec();
    }

    pub fn get_tokens(&self) -> Vec<String> {
        let e = &self.encoding.as_ref().expect("Unitialized encoding");
        return e.get_tokens().to_vec();
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

// #[no_mangle]
// pub unsafe extern "C" fn JInputSequence_from_str(str: *const c_char) -> *mut JInputSequence<'static> {
//     let cstr = unsafe { CStr::from_ptr(str).to_string_lossy().to_string() };
//     let inputSequence = Box::new(JInputSequence::from_str(cstr));
//     Box::into_raw(inputSequence)
// }

// #[no_mangle]
// pub unsafe extern "C" fn JInputSequence_from_vec_str(vec: **const c_char, len: usize) -> *mut JInputSequence {
//     let slice = unsafe { Vec::from_raw_parts(ptr, len, len) };
//     let mut v = vec![];
//
//     for elem in slice {
//         let s = CStr::from_ptr(elem).to_string_lossy().to_string();
//         v.push(s)
//     }
//     let inputSequence = Box::new(JInputSequence::from_vec_str(v));
//     Box::into_raw(inputSequence)
// }

// #[no_mangle]
// pub unsafe extern "C" fn JInputSequence_drop(p: *mut JInputSequence) {
//     Box::from_raw(p);
// }
//

//TODO: assert not null in all the pointers
#[no_mangle]
pub unsafe extern "C" fn JTokenizer_from_pretrained(identifier: *const c_char) -> *mut JTokenizer {
    let cstr = unsafe { CStr::from_ptr(identifier).to_string_lossy().to_string() };
    let boxed_a = Box::new(JTokenizer::from_pretrained(&cstr));
    Box::into_raw(boxed_a)
}

#[no_mangle]
pub unsafe extern "C" fn JTokenizer_drop(p: *mut JTokenizer) {
    Box::from_raw(p);
}

#[no_mangle]
pub unsafe extern "C" fn JTokenizer_encode_from_str(tokenizer: *mut JTokenizer, input: *const c_char) -> *mut JEncoding  {
    let instance = &*tokenizer;
    let cstr = unsafe { CStr::from_ptr(input).to_string_lossy().to_string() };
    let inputSequence = JInputSequence::from_str(cstr);
    let encodings =  Box::new(instance.encode(&inputSequence));
    return Box::into_raw(encodings);
    //println!("my tokens {:?}", e.get_tokens());
    //return e.get_ids().clone();
}

#[no_mangle]
pub unsafe extern "C" fn JTokenizer_print_tokenizer(a: *mut JTokenizer) {
    let a = &*a;
    a.print_tokenizer();
}

#[no_mangle]
pub unsafe extern "C" fn JEncoding_drop(p: *mut JEncoding) {
    Box::from_raw(p);
}

#[no_mangle]
pub unsafe extern "C" fn JEncoding_get_length(a: *mut JEncoding) -> size_t {
    let encodings = &*a;
    return  encodings.get_length();
}

#[no_mangle]
pub unsafe extern "C" fn JEncoding_get_ids(a: *mut JEncoding, buffer: *mut i64, sizeBuffer: size_t)   {

    let encodings = &*a;
    let len =  encodings.get_length();
    let vector = encodings.get_ids();
    println!("I was called in rust. tokenizer: {:?} {:?}", sizeBuffer, len);
    println!("I was called in rust. ids: {:?} ", vector);
    assert_eq!(sizeBuffer, len);
    for item in vector {
        let converted = i64::from(item);
        println!("I was called in rust. converted: {:?} ", converted);
        buffer.write(converted)
    }
}

// #[no_mangle]
// pub unsafe extern "C" fn get_ids(a: *mut JEncoding) {
//     let a = &*a;
//     a.print_tokenizer();
// }
//
//




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
