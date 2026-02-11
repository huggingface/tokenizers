use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::path::Path;
use std::fs;
use tokenizers::{Encoding, Tokenizer, AddedToken, PaddingParams, PaddingStrategy, PaddingDirection};
use serde_json::Value;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct tokenizers_encoding_t {
    pub ids: *const i32,
    pub attention_mask: *const i32,
    pub len: usize,
    pub _internal_ptr: *mut c_void,  // Store the Box pointer for cleanup
}

/// Tokenizer configuration loaded from tokenizer_config.json
/// Contains authoritative special token definitions and chat template
#[derive(Default, Clone)]
struct TokenizerConfig {
    bos_token: Option<String>,
    eos_token: Option<String>,
    pad_token: Option<String>,
    unk_token: Option<String>,
    chat_template: Option<String>,
    add_bos_token: bool,
    add_eos_token: bool,
}

impl TokenizerConfig {
    /// Load config from a directory containing tokenizer_config.json
    fn from_dir(dir: &Path) -> Option<Self> {
        let config_path = dir.join("tokenizer_config.json");
        Self::from_file(&config_path)
    }
    
    /// Load config from a specific file path
    fn from_file(path: &Path) -> Option<Self> {
        let content = fs::read_to_string(path).ok()?;
        Self::from_json(&content)
    }
    
    /// Parse config from JSON string
    fn from_json(json: &str) -> Option<Self> {
        let v: Value = serde_json::from_str(json).ok()?;
        
        // Helper to extract token string - handles both string and object formats
        let extract_token = |v: &Value, key: &str| -> Option<String> {
            match v.get(key)? {
                Value::String(s) => Some(s.clone()),
                Value::Object(obj) => obj.get("content")?.as_str().map(|s| s.to_string()),
                _ => None,
            }
        };
        
        Some(TokenizerConfig {
            bos_token: extract_token(&v, "bos_token"),
            eos_token: extract_token(&v, "eos_token"),
            pad_token: extract_token(&v, "pad_token"),
            unk_token: extract_token(&v, "unk_token"),
            chat_template: v.get("chat_template").and_then(|v| v.as_str()).map(|s| s.to_string()),
            add_bos_token: v.get("add_bos_token").and_then(|v| v.as_bool()).unwrap_or(false),
            add_eos_token: v.get("add_eos_token").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    }
    
    /// Get special token string by name
    fn get_special_token(&self, name: &str) -> Option<&str> {
        match name.to_uppercase().as_str() {
            "BOS" => self.bos_token.as_deref(),
            "EOS" => self.eos_token.as_deref(),
            "PAD" => self.pad_token.as_deref(),
            "UNK" => self.unk_token.as_deref(),
            _ => None,
        }
    }
}

/// Opaque tokenizer type exposed as void* on the C side.
/// Contains tokenizer + optional config (auto-loaded from same directory)
struct CTokenizer {
    tokenizer: Tokenizer,
    config: Option<TokenizerConfig>,
}

impl CTokenizer {
    fn new_from_file(path: &str, config_path: Option<&str>) -> Option<Self> {
        let tokenizer = Tokenizer::from_file(path).ok()?;
        // Load config: explicit path > sibling tokenizer_config.json
        let config = if let Some(cp) = config_path {
            TokenizerConfig::from_file(Path::new(cp))
        } else {
            Path::new(path).parent().and_then(TokenizerConfig::from_dir)
        };
        Some(CTokenizer { tokenizer, config })
    }
    
    fn new_from_str(json: &str) -> Option<Self> {
        let tokenizer = Tokenizer::from_bytes(json.as_bytes()).ok()?;
        // No config available when loading from string
        Some(CTokenizer { tokenizer, config: None })
    }
    
    /// Get special token ID - tries config first, falls back to heuristic
    fn get_special_token_id(&self, name: &str) -> i32 {
        // Try config first (authoritative)
        if let Some(config) = &self.config {
            if let Some(token) = config.get_special_token(name) {
                if let Some(id) = self.tokenizer.token_to_id(token) {
                    return id as i32;
                }
            }
        }
        // Fall back to heuristic
        let candidates = match name.to_uppercase().as_str() {
            "BOS" => &["<bos>", "<s>", "[CLS]", "<|begin_of_text|>", "<|startoftext|>"][..],
            "EOS" => &["<eos>", "</s>", "[SEP]", "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>"][..],
            "PAD" => &["<pad>", "[PAD]", "<|padding|>"][..],
            "UNK" => &["<unk>", "[UNK]", "<|unk|>"][..],
            _ => return -1,
        };
        for token in candidates {
            if let Some(id) = self.tokenizer.token_to_id(token) {
                return id as i32;
            }
        }
        -1
    }
}

/// Encoding data that we'll Box allocate for safe memory management
struct EncodingData {
    ids: Vec<i32>,
    attention_mask: Vec<i32>,
}

#[no_mangle]
pub extern "C" fn tokenizers_new_from_file(path: *const c_char) -> *mut c_void {
    tokenizers_new_from_file_with_config(path, ptr::null())
}

/// Create tokenizer with explicit config file path
#[no_mangle]
pub extern "C" fn tokenizers_new_from_file_with_config(
    path: *const c_char,
    config_path: *const c_char
) -> *mut c_void {
    if path.is_null() {
        return ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let config_str = if config_path.is_null() {
        None
    } else {
        let c_cfg = unsafe { CStr::from_ptr(config_path) };
        c_cfg.to_str().ok()
    };
    match CTokenizer::new_from_file(path_str, config_str) {
        Some(t) => Box::into_raw(Box::new(t)) as *mut c_void,
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_new_from_str(json: *const c_char) -> *mut c_void {
    if json.is_null() { return ptr::null_mut(); }
    let c_str = unsafe { CStr::from_ptr(json) };
    let json_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    match CTokenizer::new_from_str(json_str) {
        Some(t) => Box::into_raw(Box::new(t)) as *mut c_void,
        None => ptr::null_mut(),
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

// === Special Token IDs ===
// Unified API: automatically uses config if available, falls back to heuristic

/// Get special token ID by name ("BOS", "EOS", "PAD", "UNK")
/// Automatically uses tokenizer_config.json if found, otherwise uses heuristic.
/// Returns -1 if not found.
#[no_mangle]
pub extern "C" fn tokenizers_get_special_token_id(
    tokenizer: *mut c_void,
    name: *const c_char
) -> i32 {
    if tokenizer.is_null() || name.is_null() { return -1; }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    let c_name = unsafe { CStr::from_ptr(name) };
    let name_str = match c_name.to_str() { Ok(s) => s, Err(_) => return -1 };
    c_tok.get_special_token_id(name_str)
}

/// Get special token string by name ("BOS", "EOS", "PAD", "UNK")
/// Returns the token from config if available, otherwise null.
/// Caller must free with tokenizers_string_free.
#[no_mangle]
pub extern "C" fn tokenizers_get_special_token(
    tokenizer: *mut c_void,
    name: *const c_char
) -> *mut c_char {
    if tokenizer.is_null() || name.is_null() { return ptr::null_mut(); }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    let c_name = unsafe { CStr::from_ptr(name) };
    let name_str = match c_name.to_str() { Ok(s) => s, Err(_) => return ptr::null_mut() };
    
    if let Some(config) = &c_tok.config {
        if let Some(token) = config.get_special_token(name_str) {
            return CString::new(token).unwrap().into_raw();
        }
    }
    ptr::null_mut()
}

/// Get add_bos_token setting from config (false if no config)
#[no_mangle]
pub extern "C" fn tokenizers_get_add_bos_token(tokenizer: *mut c_void) -> bool {
    if tokenizer.is_null() { return false; }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    c_tok.config.as_ref().map_or(false, |c| c.add_bos_token)
}

/// Get add_eos_token setting from config (false if no config)
#[no_mangle]
pub extern "C" fn tokenizers_get_add_eos_token(tokenizer: *mut c_void) -> bool {
    if tokenizer.is_null() { return false; }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    c_tok.config.as_ref().map_or(false, |c| c.add_eos_token)
}

/// Check if tokenizer has a chat template (from config)
#[no_mangle]
pub extern "C" fn tokenizers_has_chat_template(tokenizer: *mut c_void) -> bool {
    if tokenizer.is_null() { return false; }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    c_tok.config.as_ref().map_or(false, |c| c.chat_template.is_some())
}

/// Get chat template string (caller must free with tokenizers_string_free)
#[no_mangle]
pub extern "C" fn tokenizers_get_chat_template(tokenizer: *mut c_void) -> *mut c_char {
    if tokenizer.is_null() { return ptr::null_mut(); }
    let c_tok = unsafe { &*(tokenizer as *mut CTokenizer) };
    if let Some(config) = &c_tok.config {
        if let Some(template) = &config.chat_template {
            return CString::new(template.as_str()).unwrap().into_raw();
        }
    }
    ptr::null_mut()
}

/// Apply a chat template to render messages
/// 
/// Arguments:
///   - tokenizer: the tokenizer instance
///   - template: Jinja2 template string
///   - messages_json: JSON array of messages with "role" and "content" fields
///   - add_generation_prompt: whether to append generation prompt
///   - bos_token: optional BOS token string
///   - eos_token: optional EOS token string
///   - error_out: pointer to error string (caller must free with tokenizers_string_free)
///
/// Returns: rendered template string (caller must free with tokenizers_string_free), or null on error
#[no_mangle]
pub extern "C" fn tokenizers_apply_chat_template(
    tokenizer: *mut c_void,
    template: *const c_char,
    messages_json: *const c_char,
    add_generation_prompt: bool,
    bos_token: *const c_char,
    eos_token: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    if tokenizer.is_null() || template.is_null() || messages_json.is_null() {
        if !error_out.is_null() {
            let err = CString::new("Invalid arguments: null pointers provided").unwrap();
            unsafe { *error_out = err.into_raw(); }
        }
        return ptr::null_mut();
    }

    let template_str = match unsafe { CStr::from_ptr(template) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let err = CString::new("Invalid template string encoding").unwrap();
                unsafe { *error_out = err.into_raw(); }
            }
            return ptr::null_mut();
        }
    };

    let messages_json_str = match unsafe { CStr::from_ptr(messages_json) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let err = CString::new("Invalid messages JSON encoding").unwrap();
                unsafe { *error_out = err.into_raw(); }
            }
            return ptr::null_mut();
        }
    };

    let bos_opt = if !bos_token.is_null() {
        match unsafe { CStr::from_ptr(bos_token) }.to_str() {
            Ok(s) => Some(s.to_string()),
            Err(_) => {
                if !error_out.is_null() {
                    let err = CString::new("Invalid BOS token encoding").unwrap();
                    unsafe { *error_out = err.into_raw(); }
                }
                return ptr::null_mut();
            }
        }
    } else {
        None
    };

    let eos_opt = if !eos_token.is_null() {
        match unsafe { CStr::from_ptr(eos_token) }.to_str() {
            Ok(s) => Some(s.to_string()),
            Err(_) => {
                if !error_out.is_null() {
                    let err = CString::new("Invalid EOS token encoding").unwrap();
                    unsafe { *error_out = err.into_raw(); }
                }
                return ptr::null_mut();
            }
        }
    } else {
        None
    };

    // Parse messages JSON
    let messages: Vec<tokenizers::Message> = match serde_json::from_str(messages_json_str) {
        Ok(msgs) => msgs,
        Err(e) => {
            if !error_out.is_null() {
                let err = CString::new(format!("Failed to parse messages JSON: {}", e)).unwrap();
                unsafe { *error_out = err.into_raw(); }
            }
            return ptr::null_mut();
        }
    };

    // Create and apply chat template
    match tokenizers::ChatTemplate::new(template_str.to_string(), bos_opt, eos_opt) {
        Ok(chat_template) => {
            let inputs = tokenizers::ChatTemplateInputs::new(messages, add_generation_prompt);
            match chat_template.apply(inputs) {
                Ok(result) => {
                    CString::new(result).unwrap().into_raw()
                }
                Err(e) => {
                    if !error_out.is_null() {
                        let err = CString::new(format!("Template rendering failed: {}", e)).unwrap();
                        unsafe { *error_out = err.into_raw(); }
                    }
                    ptr::null_mut()
                }
            }
        }
        Err(e) => {
            if !error_out.is_null() {
                let err = CString::new(format!("Failed to compile template: {}", e)).unwrap();
                unsafe { *error_out = err.into_raw(); }
            }
            ptr::null_mut()
        }
    }
}

