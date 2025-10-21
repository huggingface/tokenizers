use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;
use tokenizers::{Encoding, Tokenizer};
use tokenizers::AddedToken;

#[repr(C)]
pub struct tokenizers_encoding_t {
    pub ids: *const i32,
    pub len: usize,
}

/// Opaque tokenizer type exposed as void* on the C side.
struct CTokenizer {
    tokenizer: Tokenizer,
}

#[no_mangle]
pub extern "C" fn tokenizers_new_from_file(path: *const c_char) -> *mut c_void {
    if path.is_null() {
        return ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    match Tokenizer::from_file(path_str) {
        Ok(t) => {
            let boxed = Box::new(CTokenizer { tokenizer: t });
            Box::into_raw(boxed) as *mut c_void
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_new_from_str(json: *const c_char) -> *mut c_void {
    if json.is_null() { return ptr::null_mut(); }
    let c_str = unsafe { CStr::from_ptr(json) };
    let bytes = c_str.to_bytes();
    match Tokenizer::from_bytes(bytes) {
        Ok(t) => {
            let boxed = Box::new(CTokenizer { tokenizer: t });
            Box::into_raw(boxed) as *mut c_void
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_free(tokenizer: *mut c_void) {
    if tokenizer.is_null() { return; }
    unsafe { drop(Box::from_raw(tokenizer as *mut CTokenizer)); }
}

#[no_mangle]
pub extern "C" fn tokenizers_encode(
    tokenizer: *mut c_void,
    text: *const c_char,
    add_special_tokens: bool,
) -> tokenizers_encoding_t {
    if tokenizer.is_null() || text.is_null() {
        return tokenizers_encoding_t { ids: ptr::null(), len: 0 };
    }
    let c_tok = unsafe { &mut *(tokenizer as *mut CTokenizer) };
    let c_text = unsafe { CStr::from_ptr(text) };
    let text_str = match c_text.to_str() { Ok(s) => s, Err(_) => {
        return tokenizers_encoding_t { ids: ptr::null(), len: 0 };
    }};

    let encoding: Encoding = match c_tok.tokenizer.encode(text_str, add_special_tokens) {
        Ok(e) => e,
        Err(_) => return tokenizers_encoding_t { ids: ptr::null(), len: 0 },
    };

    let ids_vec: Vec<i32> = encoding.get_ids().iter().map(|&v| v as i32).collect();
    let len = ids_vec.len();
    let ptr_ids = ids_vec.as_ptr();
    // Leak the vec, will be reclaimed in free_encoding
    std::mem::forget(ids_vec);
    tokenizers_encoding_t { ids: ptr_ids, len }
}

#[no_mangle]
pub extern "C" fn tokenizers_free_encoding(enc: tokenizers_encoding_t) {
    if enc.ids.is_null() { return; }
    // Reconstruct Vec to drop
    unsafe { Vec::from_raw_parts(enc.ids as *mut i32, enc.len, enc.len); }
}

#[no_mangle]
pub extern "C" fn tokenizers_version() -> *const c_char {
    // Return a static C string with version info.
    static VERSION: &str = concat!("tokenizers_c ", env!("CARGO_PKG_VERSION"));
    CString::new(VERSION).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn tokenizers_string_free(s: *mut c_char) {
    if s.is_null() { return; }
    unsafe { drop(CString::from_raw(s)); }
}

#[no_mangle]
pub extern "C" fn tokenizers_vocab_size(tokenizer: *mut c_void) -> usize {
    if tokenizer.is_null() { return 0; }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    c_tok.tokenizer.get_vocab(true).len()
}

#[no_mangle]
pub extern "C" fn tokenizers_token_to_id(tokenizer: *mut c_void, token: *const c_char) -> i32 {
    if tokenizer.is_null() || token.is_null() { return -1; }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    let c_token = unsafe { CStr::from_ptr(token) };
    let token_str = match c_token.to_str() { Ok(s) => s, Err(_) => return -1 };
    match c_tok.tokenizer.token_to_id(token_str) {
        Some(id) => id as i32,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_add_special_token(tokenizer: *mut c_void, token: *const c_char) -> bool {
    if tokenizer.is_null() || token.is_null() { return false; }
    let c_tok = unsafe { &mut *(tokenizer as *mut CTokenizer) };
    let c_token = unsafe { CStr::from_ptr(token) };
    let token_str = match c_token.to_str() { Ok(s) => s, Err(_) => return false };
    let added = AddedToken::from(token_str.to_string(), true);
    c_tok.tokenizer.add_special_tokens(&[added]);
    true
}
