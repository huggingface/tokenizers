use std::ffi::CStr;
use std::ffi::CString;
use std::mem::forget;
use std::os::raw::{c_char, c_int, c_uint};
use std::str::FromStr;
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
        match config.to_str() {
            Ok(config_file) => match Tokenizer::from_file(config_file) {
                Ok(tokenizer) => return Box::into_raw(Box::new(tokenizer)),
                Err(error) => panic!("Unable to read tokenizer from file: {:?}", error),
            },
            Err(error) => panic!("Unable to read config: {:?}", error),
        }
    }
}

#[no_mangle]
pub extern "C" fn deserialize_tokenizer_from_json(cjson: *const c_char) -> *mut Tokenizer {
    unsafe {
        let json = CStr::from_ptr(cjson);
        match json.to_str() {
            Ok(json_str) => match Tokenizer::from_str(json_str) {
                Ok(tokenizer) => return Box::into_raw(Box::new(tokenizer)),
                Err(error) => panic!("Unable to read tokenizer from json: {:?}", error),
            },
            Err(error) => panic!("Unable to read json string: {:?}", error),
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
        match config.to_str() {
            Ok(config_file) => match tokenizer.save(config_file, false) {
                Ok(res) => return res,
                Err(error) => panic!("Unable to save tokenizer to file: {:?}", error),
            },
            Err(error) => panic!("Unable to read config: {:?}", error),
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
    let ids_ = ids.clone();
    match tokenizer.decode(ids, false) {
        Ok(res) => {
            let res_ = res.clone();
            match CString::new(res) {
                Ok(c_str) => {
                    let ptr = c_str.into_raw();
                    return ptr;
                }
                Err(error) => panic!("Unable to convert tokenizer result to CString: {:?} {:?} {:?}", error, res_, ids_),
            }
        },
        Err(error) => panic!("Unable to decode ids: {:?}", error),
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
        match cstring.to_str() {
            Ok(s) => {
                let token = AddedToken::from(s, true);
                tokenizer.add_special_tokens(&[token]);
            }
            Err(error) => panic!("Unable to read token: {:?}", error),
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
