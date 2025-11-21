use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;
use tokenizers::{Encoding, Tokenizer};
use tokenizers::AddedToken;

#[repr(C)]
#[derive(Copy, Clone)]
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
pub extern "C" fn tokenizers_id_to_token(tokenizer: *mut c_void, id: i32) -> *mut c_char {
    if tokenizer.is_null() { return ptr::null_mut(); }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    match c_tok.tokenizer.id_to_token(id as u32) {
        Some(token) => CString::new(token).unwrap().into_raw(),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_decode(
    tokenizer: *mut c_void,
    ids: *const i32,
    len: usize,
    skip_special_tokens: bool
) -> *mut c_char {
    if tokenizer.is_null() || ids.is_null() { return ptr::null_mut(); }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    let ids_slice_i32 = unsafe { std::slice::from_raw_parts(ids, len) };
    let ids_slice_u32: Vec<u32> = ids_slice_i32.iter().map(|&id| id as u32).collect();
    
    match c_tok.tokenizer.decode(&ids_slice_u32, skip_special_tokens) {
        Ok(s) => CString::new(s).unwrap().into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_save(tokenizer: *mut c_void, path: *const c_char, pretty: bool) -> bool {
    if tokenizer.is_null() || path.is_null() { return false; }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    let c_path = unsafe { CStr::from_ptr(path) };
    let path_str = match c_path.to_str() { Ok(s) => s, Err(_) => return false };
    
    c_tok.tokenizer.save(path_str, pretty).is_ok()
}

#[no_mangle]
pub extern "C" fn tokenizers_to_str(tokenizer: *mut c_void, pretty: bool) -> *mut c_char {
    if tokenizer.is_null() { return ptr::null_mut(); }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    match c_tok.tokenizer.to_string(pretty) {
        Ok(s) => CString::new(s).unwrap().into_raw(),
        Err(_) => ptr::null_mut(),
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

#[no_mangle]
pub extern "C" fn tokenizers_add_special_tokens(
    tokenizer: *mut c_void,
    tokens: *const *const c_char,
    len: usize
) -> usize {
    if tokenizer.is_null() || tokens.is_null() { return 0; }
    let c_tok = unsafe { &mut *(tokenizer as *mut CTokenizer) };
    let c_tokens_ptrs = unsafe { std::slice::from_raw_parts(tokens, len) };
    
    let mut added_tokens = Vec::new();
    for &ptr in c_tokens_ptrs {
        if ptr.is_null() { continue; }
        let c_str = unsafe { CStr::from_ptr(ptr) };
        if let Ok(s) = c_str.to_str() {
            added_tokens.push(AddedToken::from(s.to_string(), true));
        }
    }
    
    c_tok.tokenizer.add_special_tokens(&added_tokens)
}

#[no_mangle]
pub extern "C" fn tokenizers_encode_batch(
    tokenizer: *mut c_void,
    texts: *const *const c_char,
    len: usize,
    add_special_tokens: bool
) -> *mut tokenizers_encoding_t {
    if tokenizer.is_null() || texts.is_null() { return ptr::null_mut(); }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    let c_texts_ptrs = unsafe { std::slice::from_raw_parts(texts, len) };
    
    let mut inputs = Vec::with_capacity(len);
    for &ptr in c_texts_ptrs {
        if ptr.is_null() { continue; }
        let c_str = unsafe { CStr::from_ptr(ptr) };
        if let Ok(s) = c_str.to_str() {
            inputs.push(s);
        }
    }
    
    let encode_inputs: Vec<tokenizers::EncodeInput> = inputs.iter()
        .map(|&s| tokenizers::EncodeInput::Single(s.into()))
        .collect();

    let encodings = match c_tok.tokenizer.encode_batch(encode_inputs, add_special_tokens) {
        Ok(e) => e,
        Err(_) => return ptr::null_mut(),
    };

    let mut c_encodings = Vec::with_capacity(encodings.len());
    for encoding in encodings {
        let ids_vec: Vec<i32> = encoding.get_ids().iter().map(|&v| v as i32).collect();
        let len = ids_vec.len();
        let ptr_ids = ids_vec.as_ptr();
        std::mem::forget(ids_vec);
        c_encodings.push(tokenizers_encoding_t { ids: ptr_ids, len });
    }
    
    let ptr = c_encodings.as_mut_ptr();
    std::mem::forget(c_encodings);
    ptr
}

#[no_mangle]
pub extern "C" fn tokenizers_free_batch_encoding(encodings: *mut tokenizers_encoding_t, len: usize) {
    if encodings.is_null() { return; }
    let slice = unsafe { std::slice::from_raw_parts_mut(encodings, len) };
    for enc in slice.iter() {
        tokenizers_free_encoding(*enc);
    }
    unsafe { Vec::from_raw_parts(encodings, len, len); }
}
