use std::ffi::CStr;
use std::ffi::CString;
use std::mem::forget;
use std::os::raw::{c_char, c_int, c_uint};
use std::mem;

use tokenizers::models::bpe::BpeBuilder;
use tokenizers::models::bpe::BPE;
use tokenizers::models::unigram::*;
use tokenizers::tokenizer::Encoding;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::roberta::RobertaProcessing;


#[no_mangle]
pub extern "C" fn mk_t5_tokenizer(cvocab_file: *const c_char, ctokenizer_file: *const c_char,) -> *mut Tokenizer {
    unsafe {
        // let t = Tokenizer::new();
        unimplemented!()
    }
}

#[no_mangle]
pub extern "C" fn mk_roberta_tokenizer(
    cvocab: *const c_char,
    cmerges: *const c_char,
) -> *mut Tokenizer {
    unsafe {
        let vocab = CStr::from_ptr(cvocab);
        let merges = CStr::from_ptr(cmerges);
        if let (Ok(vocab_file), Ok(merges_file)) = (vocab.to_str(), merges.to_str()) {
        let bpe_builder = BPE::from_file(vocab_file, merges_file);
        let bpe = bpe_builder.build().unwrap();
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(ByteLevel::default());
        tokenizer.with_post_processor(RobertaProcessing::default());
        return Box::into_raw(Box::new(tokenizer));
        } else {
            panic!("Unable to read parameters.");
        }
    }
}

#[no_mangle]
pub extern "C" fn mk_tokenizer(cvocab: *const c_char, cmerges: *const c_char) -> *mut Tokenizer {
    unsafe {
        let vocab = CStr::from_ptr(cvocab);
        let merges = CStr::from_ptr(cmerges);
        if let (Ok(vocab_file), Ok(merges_file)) = (vocab.to_str(), merges.to_str()) {
            let bpe_builder = BPE::from_file(vocab_file, merges_file);
            let bpe = bpe_builder.build().unwrap();
            return Box::into_raw(Box::new(Tokenizer::new(bpe)));
        } else {
            panic!("Unable to read parameters.");
        }
    }
}

#[no_mangle]
pub extern "C" fn encode(text: *const c_char, ptr: *mut Tokenizer) -> *mut Encoding {
    unsafe {
        let cstring = CStr::from_ptr(text);
        let tokenizer = {
            assert!(!ptr.is_null());
            &mut *ptr
        };
        if let Ok(input) = cstring.to_str() {
            let encoding = tokenizer.encode(input, false).unwrap();
            return Box::into_raw(Box::new(encoding));
        } else {
            panic!("Unable to read parameters.");
        }
    }
}

#[repr(C)]
pub struct CTokens {
    length: c_int,
    data: *const *const c_char
}

#[no_mangle]
pub extern "C" fn get_tokens(ptr: *mut Encoding) -> *mut CTokens {
    unsafe {
        let encoding = {
            assert!(!ptr.is_null());
            &mut *ptr
        };
        let result = encoding.get_tokens();
        let mut cstr_vec: Vec<CString> = vec![];
        for s in result {
            let cstr = CString::new(s.as_str()).unwrap();
            cstr_vec.push(cstr);
        }
        cstr_vec.shrink_to_fit();

        let mut c_char_vec: Vec<*const c_char> = vec![];
        for s in &cstr_vec {
            let value = s.as_ptr();
            c_char_vec.push(value);
        }

        let array = CTokens { length: cstr_vec.len() as c_int, data: c_char_vec.as_ptr()};
        // todo - do this without leaking
        forget(cstr_vec);
        forget(c_char_vec);
        return Box::into_raw(Box::new(array));
    }
}

#[repr(C)]
pub struct CIDs {
    length: c_uint,
    data: *const c_uint
}

#[no_mangle]
pub extern "C" fn get_ids(ptr: *mut Encoding) -> *mut CIDs {
    unsafe {
        let encoding = {
            assert!(!ptr.is_null());
            &mut *ptr
        };
        let result = encoding.get_ids();
        forget(result);
        let array = CIDs { length: result.len() as c_uint, data: result.as_ptr()};
        return Box::into_raw(Box::new(array));
    }
}
