use std::ffi::CStr;
use std::ffi::CString;
use std::mem::forget;
use std::os::raw::{c_char, c_int, c_uint};
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::roberta::RobertaProcessing;
use tokenizers::tokenizer::Encoding;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::AddedToken;

#[no_mangle]
pub extern "C" fn deserialize_tokenizer(cconfig: *const c_char) -> *mut Tokenizer {
    unsafe {
        let config = CStr::from_ptr(cconfig);
        if let Ok(config_file) = config.to_str() {
            if let Ok(tokenizer) = Tokenizer::from_file(config_file) {
                return Box::into_raw(Box::new(tokenizer));
            } else {
                panic!("Unable to read tokenizer from file.");
            }
        } else {
            panic!("Unable to read config.");
        }
    }
}

#[no_mangle]
pub extern "C" fn serialize_tokenizer(cconfig: *const c_char, ptr: *mut Tokenizer) -> () {
    unsafe {
        let config = CStr::from_ptr(cconfig);
        let tokenizer = {
            assert!(!ptr.is_null());
            &mut *ptr
        };
        if let Ok(config_file) = config.to_str() {
            if let Ok(res) = tokenizer.save(config_file, false) {
                return res;
            } else {
                panic!("Unable to save tokenizer to file.");
            }
        } else {
            panic!("Unable to read config.");
        }
    }
}

#[no_mangle]
pub extern "C" fn free_tokenizer(ptr: *mut Tokenizer) -> () {
    unsafe { Box::from_raw(ptr) };
}

#[no_mangle]
pub extern "C" fn free_cstr(ptr: *mut c_char) -> () {
    unsafe { CString::from_raw(ptr) };
}

#[no_mangle]
pub extern "C" fn mk_wordpiece_tokenizer(cvocab: *const c_char) -> *mut Tokenizer {
    unsafe {
        let vocab = CStr::from_ptr(cvocab);
        if let Ok(vocab_file) = vocab.to_str() {
            let wp_builder = WordPiece::from_file(vocab_file);
            let wp = wp_builder.build().unwrap();
            let mut tokenizer = Tokenizer::new(wp);
            return Box::into_raw(Box::new(tokenizer));
        } else {
            panic!("Unable to read parameters.");
        }
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
            tokenizer.with_post_processor(
                RobertaProcessing::new(("</s>".to_string(), 2), ("<s>".to_string(), 0))
                    .trim_offsets(true)
                    .add_prefix_space(false),
            );
            return Box::into_raw(Box::new(tokenizer));
        } else {
            panic!("Unable to read parameters.");
        }
    }
}

#[no_mangle]
pub extern "C" fn mk_bpe_tokenizer(
    cvocab: *const c_char,
    cmerges: *const c_char,
) -> *mut Tokenizer {
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

#[no_mangle]
pub extern "C" fn decode(clength: c_uint, cids: *const c_uint, ptr: *mut Tokenizer) -> *mut c_char {
    let tokenizer = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let mut ids: Vec<u32> = vec![];
    for n in 0..clength {
        let p = unsafe { cids.offset(n as isize) };
        unsafe { ids.push(*p) };
    }
    ids.shrink_to_fit();
    if let Ok(res) = tokenizer.decode(ids, false) {
        let c_str = CString::new(res).unwrap();
        let ptr = c_str.into_raw();
        return ptr;
    } else {
        panic!("Unable to decode ids.");
    }
}

#[no_mangle]
pub extern "C" fn add_special_token(ctoken: *const c_char, ptr: *mut Tokenizer) -> () {
    unsafe {
        let cstring = CStr::from_ptr(ctoken);
        let tokenizer = {
            assert!(!ptr.is_null());
            &mut *ptr
        };
        if let Ok(s) = cstring.to_str() {
            let token = AddedToken::from(s, true);
            tokenizer.add_special_tokens(&[token]);
        } else {
            panic!("Unable to read token.");
        }
    }
}

#[repr(C)]
pub struct CTokens {
    length: c_int,
    data: *const *const c_char,
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

        let array = CTokens {
            length: cstr_vec.len() as c_int,
            data: c_char_vec.as_ptr(),
        };
        // todo - do this without leaking
        forget(cstr_vec);
        forget(c_char_vec);
        return Box::into_raw(Box::new(array));
    }
}

#[repr(C)]
pub struct CIDs {
    length: c_uint,
    data: *const c_uint,
}

#[no_mangle]
pub extern "C" fn get_ids(ptr: *mut Encoding) -> *mut CIDs {
    unsafe {
        let encoding = {
            assert!(!ptr.is_null());
            &mut *ptr
        };
        let mut result = encoding.get_ids();
        /*
        println!("rust encoding:");
        for id in result {
            println!("{} ", id);
        }
        */
        // forget(result);
        let mut array = CIDs {
            length: result.len() as c_uint,
            data: result.as_ptr(),
        };
        return Box::into_raw(Box::new(array));
    }
}
