use std::ffi::c_char;

use crate::decoded_string::DecodedString;
use crate::encoding::Encoding;
use crate::error::Error;
use crate::options::{DecodeOptions, EncodeOptions};
use crate::utils::{RustOwned, Slice, convert_c_buf, convert_c_str, free_ptr, wrap_in_ptr};
use tk_encode::pipeline::PipelineTokenizer;

pub struct Tokenizer(PipelineTokenizer);
impl RustOwned for Tokenizer {}

/// Instantiates a tokenizer from its JSON config file and writes its pointer to `out`, or NULL
/// to `out` if this call fails.
///
/// # Safety
/// 1. `path` must be NULL, or a NUL-terminated string that is neither freed nor modified while
///    this call runs.
/// 2. `out` must be NULL, or a writable pointer to a `TkTokenizer *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_from_file(
    path: *const c_char,
    out: *mut *mut Tokenizer,
) -> *mut Error {
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
    unsafe { wrap_in_ptr(out, body) }
}

/// Frees `tokenizer` and writes NULL to it, so calling this again on the same pointer is a
/// no-op instead of a double free.
///
/// # Safety
/// `tokenizer` must be NULL, or point to a writable `TkTokenizer *` holding NULL or a tokenizer
/// that is not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_free(tokenizer: *mut *mut Tokenizer) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_ptr(tokenizer) }
}

/// Encodes `input` to tokens with `tokenizer` and writes the encoding's pointer to `out`, or
/// NULL to `out` if this call fails. NULL `options` means the defaults.
///
/// # Safety
/// 1. `tokenizer` must be NULL, or a tokenizer that is not freed while this call runs.
/// 2. `input` must be NULL, or point to `input_len` readable bytes that are neither freed nor
///    modified while this call runs.
/// 3. `options` must be NULL, or encode options that are neither freed nor modified
///    (`tk_encode_options_set_*`) while this call runs.
/// 4. `out` must be NULL, or a writable pointer to a `TkEncoding *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_encode(
    tokenizer: Option<&Tokenizer>,
    input: *const c_char,
    input_len: usize,
    options: Option<&EncodeOptions>,
    out: *mut *mut Encoding,
) -> *mut Error {
    let inner = move || -> tk_encode::Result<Encoding> {
        let Some(tokenizer) = tokenizer else {
            return Err("tokenizer must not be NULL".into());
        };
        if input.is_null() {
            return Err("input must not be NULL".into());
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let input = unsafe { convert_c_buf(input, input_len) }?;
        let options = options.copied().unwrap_or_default();
        let mut encodings = tokenizer
            .0
            .encode(input, options.add_special_tokens)
            .wait()?;
        // `Inputs::Single` always yields exactly one result.
        let encoding = encodings.remove(0);
        let ids = encoding.ids().iter().map(|id| id.id()).collect();
        let type_ids = encoding.type_ids().map(<[u8]>::to_vec);
        Ok(Encoding { ids, type_ids })
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_ptr(out, inner) }
}

/// Decodes `ids` back into a UTF-8 string with `tokenizer` and writes the string's pointer to
/// `out`, or NULL to `out` if this call fails. NULL `options` means the defaults.
///
/// # Safety
/// 1. `tokenizer` must be NULL, or a tokenizer that is not freed while this call runs.
/// 2. `ids.ptr` must be NULL, or point to `ids.len` readable `uint32_t`s that are neither freed
///    nor modified while this call runs.
/// 3. `options` must be NULL, or decode options that are neither freed nor modified
///    (`tk_decode_options_set_*`) while this call runs.
/// 4. `out` must be NULL, or a writable pointer to a `TkDecodedString *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_decode(
    tokenizer: Option<&Tokenizer>,
    ids: Slice<u32>,
    options: Option<&DecodeOptions>,
    out: *mut *mut DecodedString,
) -> *mut Error {
    let inner = move || -> tk_encode::Result<DecodedString> {
        let Some(tokenizer) = tokenizer else {
            return Err("tokenizer must not be NULL".into());
        };
        // SAFETY: caller's obligation, documented above.
        let ids = unsafe { ids.as_slice() };
        let options = options.copied().unwrap_or_default();
        let decoded = tokenizer.0.decode(ids, options.skip_special_tokens)?;
        Ok(DecodedString(decoded))
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_ptr(out, inner) }
}
