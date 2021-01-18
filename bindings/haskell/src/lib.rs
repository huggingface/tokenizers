use std::ffi::CStr;
use std::ffi::CString;
use std::mem::forget;
use std::os::raw::c_char;

use tokenizers::models::bpe::BpeBuilder;
use tokenizers::models::bpe::BPE;
use tokenizers::tokenizer::Encoding;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::roberta::RobertaProcessing;


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
pub extern "C" fn mk_bpe_builder_from_files(
    cvocab: *const c_char,
    cmerges: *const c_char,
) -> *mut BpeBuilder {
    unsafe {
        let vocab = CStr::from_ptr(cvocab);
        let merges = CStr::from_ptr(cmerges);
        if let (Ok(vocab_file), Ok(merges_file)) = (vocab.to_str(), merges.to_str()) {
            Box::into_raw(Box::new(BPE::from_file(vocab_file, merges_file)))
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

// https://users.rust-lang.org/t/solved-how-to-export-vec-string-to-c-with-ffi/7121
#[repr(C)]
pub struct Array {
    data: *const *const c_char
}

#[no_mangle]
pub extern "C" fn get_tokens(ptr: *mut Encoding) -> *const *const c_char {
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
            forget(value);
            c_char_vec.push(value);
        }

        let array = c_char_vec.as_ptr();
        // todo - do this without leaking
        forget(cstr_vec);
        forget(c_char_vec);
        return array;
    }
}

#[no_mangle]
pub extern "C" fn tokenize(text: *const c_char, ptr: *mut Tokenizer) {
    unsafe {
        let cstring = CStr::from_ptr(text);
        let tokenizer = {
            assert!(!ptr.is_null());
            &mut *ptr
        };
        if let Ok(input) = cstring.to_str() {
            let encoding = tokenizer.encode(input, false).unwrap();
            println!("{:?}", encoding.get_tokens());
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenize_test(x: *const c_char) {
    unsafe {
        let cstring = CStr::from_ptr(x);
        if let Ok(input) = cstring.to_str() {
            let bpe_builder = BPE::from_file("roberta-base-vocab.json", "roberta-base-merges.txt");
            let bpe = bpe_builder.dropout(0.1).build().unwrap();
            let tokenizer = Tokenizer::new(bpe);
            let encoding = tokenizer.encode(input, false).unwrap();
            println!("{:?}", encoding.get_tokens());
        } else {
            panic!("Unable to read parameter.");
        }
    }
}
