use std::ffi::c_char;

use crate::error::{TkError, catch_panic};
use crate::utils::{
    RustOwned, TkHandle, TkSlice, convert_c_str, drop_tk_handle, write_c_slice, write_tk_handle,
};
use tk_encode::pipeline::PipelineTokenizer;

pub struct TkTokenizer(PipelineTokenizer);
impl RustOwned for TkTokenizer {}

pub struct TkEncoding {
    ids: Vec<u32>,
    #[allow(unused)]
    type_ids: Option<Vec<u8>>,
}
impl RustOwned for TkEncoding {}

/// Instantiates a tokenizer from its JSON config file
///
/// # Safety
/// 1. [`path`] must be non-NULL, point to a NUL-terminated byte string, and be valid for reads
///    up to and including that NUL byte for the duration of this call. It must not be mutated
///    while this function runs.
/// 2. [`out`] must be valid, writeable pointer to a [`TkHandle<TkTokenizer>`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_from_file(
    path: *const c_char,
    out: *mut TkHandle<TkTokenizer>,
) -> TkHandle<TkError> {
    catch_panic(move || -> tk_encode::Result<()> {
        // SAFETY: caller's obligation, documented above.
        let path = unsafe { convert_c_str(path) }?;
        let canonical = tk_convert::canonicalize_file(path)?;
        let tokenizer = tk_serialize::from_json(&canonical)?;
        // SAFETY: caller's obligation, documented above.
        unsafe { write_tk_handle(out, TkTokenizer(tokenizer)) };
        Ok(())
    })
}
/// Frees memory of the TkTokenizer
///
/// # Safety
/// `handle` must either be NULL or a not-freed-yet [`TkHandle<TkTokenizer>`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_free(
    tk_tokenizer: TkHandle<TkTokenizer>,
) -> TkHandle<TkError> {
    catch_panic(move || -> Result<(), std::convert::Infallible> {
        // SAFETY: caller's obligation, documented above.
        unsafe { drop_tk_handle(tk_tokenizer) };
        Ok(())
    })
}

/// Encodes the input text to tokens using the provided tokenizer.
///
/// # Safety
/// 1. [`tokenizer`] must be non-NULL and point to a valid, not-yet-freed [`TkHandle<TkTokenizer>`].
/// 2. [`input`] must be non-NULL, point to a NUL-terminated byte string, and be valid for reads
///    up to and including that NUL byte for the duration of this call. It must not be mutated
///    while this function runs.
/// 3. [`out`] must be a valid, writeable pointer to a [`TkHandle<TkEncoding>`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_tokenizer_encode(
    tk_tokenizer: TkHandle<TkTokenizer>,
    input: *const c_char,
    add_special_tokens: bool,
    out: *mut TkHandle<TkEncoding>,
) -> TkHandle<TkError> {
    catch_panic(move || -> tk_encode::Result<()> {
        // SAFETY: caller's obligation, documented above.
        let input = unsafe { convert_c_str(input) }?;
        // SAFETY: caller's obligation, documented above.
        let tokenizer = unsafe { tk_tokenizer.as_ref() };
        let mut encodings = tokenizer.0.encode(input, add_special_tokens).wait()?;
        // `Inputs::Single` always yields exactly one result.
        let encoding = encodings.remove(0);
        let ids = encoding.ids().iter().map(|id| id.id()).collect();
        let type_ids = encoding.type_ids().map(<[u8]>::to_vec);
        // SAFETY: caller's obligation, documented above.
        unsafe { write_tk_handle(out, TkEncoding { ids, type_ids }) };
        Ok(())
    })
}

/// Frees memory backing the [`TkEncoding`]
///
/// # Safety
/// `handle` must either be NULL or a not-freed-yet [`TkHandle<TkTokenizer>`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_free(tk_encoding: TkHandle<TkEncoding>) -> TkHandle<TkError> {
    catch_panic(move || -> Result<(), std::convert::Infallible> {
        // SAFETY: caller's obligation, documented above.
        unsafe { drop_tk_handle(tk_encoding) };
        Ok(())
    })
}

/// Writes `tk_encoding`'s token ids to `out`.
///
/// # Safety
/// 1. [`tk_encoding`] must be non-NULL and point to a valid, not-yet-freed [`TkHandle<TkEncoding>`].
/// 2. [`out`] must be a valid, writeable pointer to a [`TkSlice<u32>`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_ids(
    tk_encoding: TkHandle<TkEncoding>,
    out: *mut TkSlice<u32>,
) -> TkHandle<TkError> {
    catch_panic(move || -> Result<(), std::convert::Infallible> {
        // SAFETY: caller's obligation, documented above.
        let tk_encoding = unsafe { tk_encoding.as_ref() };
        // SAFETY: caller's obligation, documented above.
        unsafe { write_c_slice(out, &tk_encoding.ids) };
        Ok(())
    })
}
