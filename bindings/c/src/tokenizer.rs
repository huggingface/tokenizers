use std::ffi::c_char;

use crate::decoded_string::DecodedString;
use crate::encoding::Encoding;
use crate::error::Error;
use crate::utils::{
    Handle, RustOwned, Slice, convert_c_buf, convert_c_str, free_handle, wrap_in_handle,
};
use tk_encode::pipeline::PipelineTokenizer;

pub struct Tokenizer(PipelineTokenizer);
impl RustOwned for Tokenizer {}

/// Instantiates a tokenizer from its JSON config file.
///
/// # Safety
/// 1. `path` must be non-NULL, point to a NUL-terminated byte string, and be valid for reads
///    up to and including that NUL byte for the duration of this call. It must not be mutated
///    while this function runs.
/// 2. `out` must be valid, writable pointer to a `TkHandle_Tokenizer`. On return, it
///    holds a live handle if this function returns NULL, or NULL otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_from_file(
    path: *const c_char,
    out: *mut Handle<Tokenizer>,
) -> Handle<Error> {
    let body = move || -> tk_encode::Result<Tokenizer> {
        if path.is_null() {
            return Err("path must not be NULL".into());
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let path = unsafe { convert_c_str(path) }?;
        let canonical = tk_convert::canonicalize_file(path)?;
        let tokenizer = tk_serialize::from_json(&canonical)?;
        Ok(Tokenizer(tokenizer))
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_handle(out, body) }
}

/// Frees `tokenizer` and writes NULL to it, so calling this again on the same pointer is a
/// no-op instead of a double free.
///
/// # Safety
/// `tokenizer` must be non-NULL and point to a `TkHandle_Tokenizer` that is either NULL
/// or live (not already freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_free(tokenizer: *mut Handle<Tokenizer>) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_handle(tokenizer) }
}

/// Encodes the input text to tokens using the provided tokenizer.
///
/// # Safety
/// 1. `tokenizer` must be non-NULL and point to a valid, not-yet-freed `TkHandle_Tokenizer`.
/// 2. `input` must be non-NULL and valid for reads of `input_len` bytes for the duration of
///    this call. It must not be mutated while this function runs.
/// 3. `out` must be a valid, writable pointer to a `TkHandle_Encoding`. On return, it
///    holds a live handle if this function returns NULL, or NULL otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_encode(
    tokenizer: Handle<Tokenizer>,
    input: *const c_char,
    input_len: usize,
    add_special_tokens: bool,
    out: *mut Handle<Encoding>,
) -> Handle<Error> {
    let inner = move || -> tk_encode::Result<Encoding> {
        if tokenizer.is_null() {
            return Err("tokenizer must not be NULL".into());
        }
        if input.is_null() {
            return Err("input must not be NULL".into());
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let input = unsafe { convert_c_buf(input, input_len) }?;
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let tokenizer = unsafe { tokenizer.as_ref() };
        let mut encodings = tokenizer.0.encode(input, add_special_tokens).wait()?;
        // `Inputs::Single` always yields exactly one result.
        let encoding = encodings.remove(0);
        let ids = encoding.ids().iter().map(|id| id.id()).collect();
        let type_ids = encoding.type_ids().map(<[u8]>::to_vec);
        Ok(Encoding { ids, type_ids })
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_handle(out, inner) }
}

/// Decodes a slice of token ids back into a utf8 string
///
/// # Safety
/// 1. `tokenizer` must be non-NULL and point to a valid, not-yet-freed `TkHandle_Tokenizer`.
/// 2. `ids.ptr` must be NULL, or valid for reads of `ids.len` elements of `u32` for the
///    duration of this call. It must not be mutated while this function runs.
/// 3. `out` must be a valid, writable pointer to a `TkHandle_DecodedString`. On return, it
///    holds a live handle if this function returns NULL, or NULL otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_decode(
    tokenizer: Handle<Tokenizer>,
    ids: Slice<u32>,
    skip_special_tokens: bool,
    out: *mut Handle<DecodedString>,
) -> Handle<Error> {
    let inner = move || -> tk_encode::Result<DecodedString> {
        if tokenizer.is_null() {
            return Err("tokenizer must not be NULL".into());
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let tokenizer = unsafe { tokenizer.as_ref() };
        // SAFETY: caller's obligation, documented above.
        let ids = unsafe { ids.as_slice() };
        let decoded = tokenizer.0.decode(ids, skip_special_tokens)?;
        Ok(DecodedString(decoded))
    };
    unsafe { wrap_in_handle(out, inner) }
}
