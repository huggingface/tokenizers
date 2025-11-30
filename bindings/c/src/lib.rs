use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;
use tokenizers::{Encoding, Tokenizer, AddedToken, PaddingParams, PaddingStrategy, PaddingDirection};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct tokenizers_encoding_t {
    pub ids: *const i32,
    pub attention_mask: *const i32,
    pub len: usize,
    pub _internal_ptr: *mut c_void,  // Store the Box pointer for cleanup
}

/// Opaque tokenizer type exposed as void* on the C side.
struct CTokenizer {
    tokenizer: Tokenizer,
}

/// Encoding data that we'll Box allocate for safe memory management
struct EncodingData {
    ids: Vec<i32>,
    attention_mask: Vec<i32>,
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
        return tokenizers_encoding_t { 
            ids: ptr::null(), 
            attention_mask: ptr::null(), 
            len: 0, 
            _internal_ptr: ptr::null_mut() 
        };
    }
    let c_tok = unsafe { &mut *(tokenizer as *mut CTokenizer) };
    let c_text = unsafe { CStr::from_ptr(text) };
    let text_str = match c_text.to_str() { Ok(s) => s, Err(_) => {
        return tokenizers_encoding_t { 
            ids: ptr::null(), 
            attention_mask: ptr::null(), 
            len: 0, 
            _internal_ptr: ptr::null_mut() 
        };
    }};

    let encoding: Encoding = match c_tok.tokenizer.encode(text_str, add_special_tokens) {
        Ok(e) => e,
        Err(_) => return tokenizers_encoding_t { 
            ids: ptr::null(), 
            attention_mask: ptr::null(), 
            len: 0, 
            _internal_ptr: ptr::null_mut() 
        },
    };

    let ids_vec: Vec<i32> = encoding.get_ids().iter().map(|&v| v as i32).collect();
    let mask_vec: Vec<i32> = encoding.get_attention_mask().iter().map(|&v| v as i32).collect();
    let len = ids_vec.len();
    
    // Allocate EncodingData on the heap using Box
    let encoding_data = Box::new(EncodingData {
        ids: ids_vec,
        attention_mask: mask_vec,
    });
    
    let ptr_ids = encoding_data.ids.as_ptr();
    let ptr_mask = encoding_data.attention_mask.as_ptr();
    
    // Convert Box to raw pointer - this transfers ownership to C
    let raw_ptr = Box::into_raw(encoding_data);
    
    tokenizers_encoding_t { 
        ids: ptr_ids, 
        attention_mask: ptr_mask, 
        len,
        _internal_ptr: raw_ptr as *mut c_void
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encode_batch(
    tokenizer: *mut c_void,
    texts: *const *const c_char,
    len: usize,
    add_special_tokens: bool,
) -> *mut tokenizers_encoding_t {
    if tokenizer.is_null() || texts.is_null() { return ptr::null_mut(); }
    let c_tok = unsafe { &mut *(tokenizer as *mut CTokenizer) };
    let c_texts_ptrs = unsafe { std::slice::from_raw_parts(texts, len) };
    
    let mut rs_texts = Vec::new();
    for &ptr in c_texts_ptrs {
        if ptr.is_null() { continue; }
        let c_str = unsafe { CStr::from_ptr(ptr) };
        if let Ok(s) = c_str.to_str() {
            rs_texts.push(s);
        }
    }
    
    let encodings = match c_tok.tokenizer.encode_batch(rs_texts, add_special_tokens) {
        Ok(e) => e,
        Err(_) => return ptr::null_mut(),
    };
    
    let mut c_encodings = Vec::with_capacity(encodings.len());
    for encoding in encodings {
        let ids_vec: Vec<i32> = encoding.get_ids().iter().map(|&v| v as i32).collect();
        let mask_vec: Vec<i32> = encoding.get_attention_mask().iter().map(|&v| v as i32).collect();
        let len = ids_vec.len();
        let ptr_ids = ids_vec.as_ptr();
        let ptr_mask = mask_vec.as_ptr();
        
        std::mem::forget(ids_vec);
        std::mem::forget(mask_vec);
        
        c_encodings.push(tokenizers_encoding_t { 
            ids: ptr_ids, 
            attention_mask: ptr_mask, 
            len,
            _internal_ptr: ptr::null_mut()  // Batch encoding has memory management issues - we'll leak for now
        });
    }
    
    let ptr = c_encodings.as_mut_ptr();
    std::mem::forget(c_encodings);
    ptr
}

#[no_mangle]
pub extern "C" fn tokenizers_free_encoding(enc: tokenizers_encoding_t) {
    if !enc._internal_ptr.is_null() {
        unsafe {
            // Reconstruct the Box from the raw pointer and let it drop naturally
            let _boxed = Box::from_raw(enc._internal_ptr as *mut EncodingData);
            // Box will be automatically dropped here, cleaning up the memory
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_free_batch_encoding(encodings: *mut tokenizers_encoding_t, len: usize) {
    if encodings.is_null() { return; }
    let slice = unsafe { std::slice::from_raw_parts_mut(encodings, len) };
    for enc in slice {
        tokenizers_free_encoding(*enc);
    }
    unsafe { Vec::from_raw_parts(encodings, len, len); }
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
pub extern "C" fn tokenizers_decode_batch(
    tokenizer: *mut c_void,
    ids: *const *const i32,
    lens: *const usize,
    batch_len: usize,
    skip_special_tokens: bool
) -> *mut *mut c_char {
    if tokenizer.is_null() || ids.is_null() || lens.is_null() { return ptr::null_mut(); }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    
    let ids_ptrs = unsafe { std::slice::from_raw_parts(ids, batch_len) };
    let lens_slice = unsafe { std::slice::from_raw_parts(lens, batch_len) };
    
    let mut batch_ids_u32 = Vec::with_capacity(batch_len);
    for i in 0..batch_len {
        let len = lens_slice[i];
        let ptr = ids_ptrs[i];
        if ptr.is_null() {
            batch_ids_u32.push(vec![]);
            continue;
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        batch_ids_u32.push(slice.iter().map(|&id| id as u32).collect());
    }
    
    let batch_ids_refs: Vec<&[u32]> = batch_ids_u32.iter().map(|v| v.as_slice()).collect();
    
    let decoded = match c_tok.tokenizer.decode_batch(&batch_ids_refs, skip_special_tokens) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    
    let mut c_strings = Vec::with_capacity(decoded.len());
    for s in decoded {
        c_strings.push(CString::new(s).unwrap().into_raw());
    }
    
    let ptr = c_strings.as_mut_ptr();
    std::mem::forget(c_strings);
    ptr
}

#[no_mangle]
pub extern "C" fn tokenizers_free_batch_decode(strings: *mut *mut c_char, len: usize) {
    if strings.is_null() { return; }
    let slice = unsafe { std::slice::from_raw_parts_mut(strings, len) };
    for &mut s in slice {
        tokenizers_string_free(s);
    }
    unsafe { Vec::from_raw_parts(strings, len, len); }
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
pub extern "C" fn tokenizers_add_tokens(
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
            added_tokens.push(AddedToken::from(s.to_string(), false));
        }
    }
    
    c_tok.tokenizer.add_tokens(&added_tokens)
}

#[repr(C)]
pub struct tokenizers_truncation_params_t {
    pub max_length: usize,
    pub stride: usize,
    pub strategy: i32, // 0: LongestFirst, 1: OnlyFirst, 2: OnlySecond
    pub direction: i32, // 0: Left, 1: Right
}

#[no_mangle]
pub extern "C" fn tokenizers_set_truncation(
    tokenizer: *mut c_void,
    params: *const tokenizers_truncation_params_t
) {
    if tokenizer.is_null() { return; }
    let c_tok = unsafe { &mut *(tokenizer as *mut CTokenizer) };
    
    if params.is_null() {
        let _ = c_tok.tokenizer.with_truncation(None);
        return;
    }
    
    let p = unsafe { &*params };
    
    let strategy = match p.strategy {
        1 => tokenizers::TruncationStrategy::OnlyFirst,
        2 => tokenizers::TruncationStrategy::OnlySecond,
        _ => tokenizers::TruncationStrategy::LongestFirst,
    };
    
    let direction = match p.direction {
        1 => tokenizers::TruncationDirection::Right,
        _ => tokenizers::TruncationDirection::Left,
    };
    
    let params = tokenizers::TruncationParams {
        max_length: p.max_length,
        stride: p.stride,
        strategy,
        direction,
    };
    
    let _ = c_tok.tokenizer.with_truncation(Some(params));
}

#[repr(C)]
pub struct tokenizers_padding_params_t {
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: *const c_char,
    pub strategy: i32, // 0: BatchLongest, 1: Fixed
    pub fixed_length: usize,
    pub direction: i32, // 0: Left, 1: Right
    pub pad_to_multiple_of: usize,
}

#[no_mangle]
pub extern "C" fn tokenizers_set_padding(
    tokenizer: *mut c_void,
    params: *const tokenizers_padding_params_t
) {
    if tokenizer.is_null() { return; }
    let c_tok = unsafe { &mut *(tokenizer as *mut CTokenizer) };
    
    if params.is_null() {
        c_tok.tokenizer.with_padding(None);
        return;
    }
    
    let p = unsafe { &*params };
    let pad_token = unsafe { CStr::from_ptr(p.pad_token) }.to_string_lossy().into_owned();
    
    let strategy = match p.strategy {
        1 => PaddingStrategy::Fixed(p.fixed_length),
        _ => PaddingStrategy::BatchLongest,
    };
    
    let direction = match p.direction {
        1 => PaddingDirection::Right,
        _ => PaddingDirection::Left,
    };
    
    let params = PaddingParams {
        strategy,
        direction,
        pad_id: p.pad_id,
        pad_type_id: p.pad_type_id,
        pad_token,
        pad_to_multiple_of: if p.pad_to_multiple_of == 0 { None } else { Some(p.pad_to_multiple_of) },
    };
    
    c_tok.tokenizer.with_padding(Some(params));
}
