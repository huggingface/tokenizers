use std::ffi::c_char;

use crate::error::TkError;
use crate::utils::{
    RustOwned, TkHandle, TkSlice, convert_c_buf, convert_c_str, free_tk_handle, wrap_in_tk_handle,
    wrap_in_tk_slice,
};
use tk_encode::pipeline::PipelineTokenizer;

pub struct TkTokenizer(PipelineTokenizer);
impl RustOwned for TkTokenizer {}

pub struct TkEncoding {
    ids: Vec<u32>,
    type_ids: Option<Vec<u8>>,
}
impl RustOwned for TkEncoding {}

/// Instantiates a tokenizer from its JSON config file.
///
/// # Safety
/// 1. `path` must be non-NULL, point to a NUL-terminated byte string, and be valid for reads
///    up to and including that NUL byte for the duration of this call. It must not be mutated
///    while this function runs.
/// 2. `out` must be valid, writable pointer to a [`TkHandle<TkTokenizer>`]. On return, it
///    holds a live handle if this function returns NULL, or NULL otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_from_file(
    path: *const c_char,
    out: *mut TkHandle<TkTokenizer>,
) -> TkHandle<TkError> {
    let body = move || -> tk_encode::Result<TkTokenizer> {
        if path.is_null() {
            return Err("path must not be NULL".into());
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let path = unsafe { convert_c_str(path) }?;
        let canonical = tk_convert::canonicalize_file(path)?;
        let tokenizer = tk_serialize::from_json(&canonical)?;
        Ok(TkTokenizer(tokenizer))
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_tk_handle(out, body) }
}

/// Frees `tk_tokenizer` and writes NULL to it, so calling this again on the same pointer is a
/// no-op instead of a double free.
///
/// # Safety
/// `tk_tokenizer` must be non-NULL and point to a [`TkHandle<TkTokenizer>`] that is either NULL
/// or live (not already freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_free(tk_tokenizer: *mut TkHandle<TkTokenizer>) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_tk_handle(tk_tokenizer) }
}

/// Encodes the input text to tokens using the provided tokenizer.
///
/// # Safety
/// 1. `tk_tokenizer` must be non-NULL and point to a valid, not-yet-freed [`TkHandle<TkTokenizer>`].
/// 2. `input` must be non-NULL and valid for reads of `input_len` bytes for the duration of
///    this call. It must not be mutated while this function runs.
/// 3. `out` must be a valid, writable pointer to a [`TkHandle<TkEncoding>`]. On return, it
///    holds a live handle if this function returns NULL, or NULL otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_encode(
    tk_tokenizer: TkHandle<TkTokenizer>,
    input: *const c_char,
    input_len: usize,
    add_special_tokens: bool,
    out: *mut TkHandle<TkEncoding>,
) -> TkHandle<TkError> {
    let inner = move || -> tk_encode::Result<TkEncoding> {
        if tk_tokenizer.is_null() {
            return Err("tk_tokenizer must not be NULL".into());
        }
        if input.is_null() {
            return Err("input must not be NULL".into());
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let input = unsafe { convert_c_buf(input, input_len) }?;
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let tokenizer = unsafe { tk_tokenizer.as_ref() };
        let mut encodings = tokenizer.0.encode(input, add_special_tokens).wait()?;
        // `Inputs::Single` always yields exactly one result.
        let encoding = encodings.remove(0);
        let ids = encoding.ids().iter().map(|id| id.id()).collect();
        let type_ids = encoding.type_ids().map(<[u8]>::to_vec);
        Ok(TkEncoding { ids, type_ids })
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_tk_handle(out, inner) }
}

/// Frees `tk_encoding` and writes NULL to it, so calling this again on the same pointer is a
/// no-op instead of a double free.
///
/// # Safety
/// `tk_encoding` must be non-NULL and point to a [`TkHandle<TkEncoding>`] that is either NULL
/// or live (not already freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_free(tk_encoding: *mut TkHandle<TkEncoding>) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_tk_handle(tk_encoding) }
}

/// Writes `tk_encoding`'s token ids to `out`.
///
/// # Safety
/// 1. `tk_encoding` must be non-NULL and point to a valid, not-yet-freed [`TkHandle<TkEncoding>`].
/// 2. `out` must be a valid, writable pointer to a [`TkSlice<u32>`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_ids(
    tk_encoding: TkHandle<TkEncoding>,
    out: *mut TkSlice<u32>,
) -> TkHandle<TkError> {
    let inner = move || -> Result<&[u32], &'static str> {
        if tk_encoding.is_null() {
            return Err("tk_encoding must not be NULL");
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let tk_encoding = unsafe { tk_encoding.as_ref() };
        Ok(&tk_encoding.ids[..])
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_tk_slice(out, inner) }
}

/// Writes `tk_encoding`'s per-token type ids to `out`, or an empty slice if the tokenizer
/// doesn't produce them.
///
/// # Safety
/// 1. `tk_encoding` must be non-NULL and point to a valid, not-yet-freed [`TkHandle<TkEncoding>`].
/// 2. `out` must be a valid, writable pointer to a [`TkSlice<u8>`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_type_ids(
    tk_encoding: TkHandle<TkEncoding>,
    out: *mut TkSlice<u8>,
) -> TkHandle<TkError> {
    let inner = move || -> Result<&[u8], &'static str> {
        if tk_encoding.is_null() {
            return Err("tk_encoding must not be NULL");
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let tk_encoding = unsafe { tk_encoding.as_ref() };
        Ok(tk_encoding.type_ids.as_deref().unwrap_or(&[]))
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_tk_slice(out, inner) }
}
