extern crate libc;
extern crate tokenizers as tk;

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr::null_mut;
use std::fmt;

use tk::Tokenizer;

use libc::boolean_t;
use std::ops::Deref;
use tk::FromPretrainedParameters;

#[repr(C)]
pub struct JTokenizer {
    tokenizer: *mut Tokenizer
}

impl JTokenizer  {

    //identifier string
    //FromPretrainedParameters: two Option of Strings
    //return type: struct result
    pub fn from_pretrained(identifier: &str) -> JTokenizer {
        let parameters = FromPretrainedParameters::default();
        let tokenizer = Tokenizer::from_pretrained(identifier, Some(parameters));
        match tokenizer {
            Ok(value) => {
                let boxed_tokenizer = Box::new(value);
                let raw = Box::into_raw(boxed_tokenizer);
                return JTokenizer{tokenizer: raw};
            }
            Err(error) => {
                println!("Problem instantiating tokenizer {:?}", error);
                return JTokenizer{tokenizer: null_mut()};
            }
        }
    }

    pub fn print_tokenizer(&self)  {
        let instance = unsafe{ &*(self.tokenizer) };
        let tokenizer = instance.to_string(true);
        match tokenizer {
            Ok(value) => {
                println!("I was called in rust. tokenizer: {:?}", value);
            }
            Err(error) => {
                println!("I was called in rust. vector: {:?}", error);
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
#[no_mangle]
pub unsafe extern "C" fn JTokenizer_from_pretrained(identifier: *mut c_char) -> *mut JTokenizer {
    let cstr = unsafe { CString::from_raw(identifier).to_string_lossy().to_string() };
    let boxed_a = Box::new(JTokenizer::from_pretrained( &cstr) );
    Box::into_raw(boxed_a)
}

#[no_mangle]
pub unsafe extern "C" fn JTokenizer_drop(p: *mut JTokenizer) {
    Box::from_raw(p);
}

#[no_mangle]
pub unsafe extern "C" fn JTokenizer_print_tokenizer(a: *mut JTokenizer) {
    let a = &*a;
    a.print_tokenizer()
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