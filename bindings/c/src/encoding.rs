use crate::error::Error;
use crate::utils::{Handle, RustOwned, Slice, free_handle, wrap_in_slice};

pub struct Encoding {
    pub(crate) ids: Vec<u32>,
    pub(crate) type_ids: Option<Vec<u8>>,
}
impl RustOwned for Encoding {}

/// Frees `encoding` and writes NULL to it, so calling this again on the same pointer is a
/// no-op instead of a double free.
///
/// # Safety
/// `encoding` must be NULL, or point to a `TkHandle_Encoding` holding NULL or a handle that is
/// not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_free(encoding: *mut Handle<Encoding>) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_handle(encoding) }
}

/// Writes `encoding`'s token ids to `out`. The slice points into `encoding` and is valid until
/// `tk_encoding_free`.
///
/// # Safety
/// 1. `encoding` must be NULL, or a `TkHandle_Encoding` that is not freed while this call runs.
/// 2. `out` must be NULL, or a writable pointer to a `TkSlice_u32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_ids(
    encoding: Handle<Encoding>,
    out: *mut Slice<u32>,
) -> Handle<Error> {
    let inner = move || -> Result<&[u32], &'static str> {
        if encoding.is_null() {
            return Err("encoding must not be NULL");
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let encoding = unsafe { encoding.as_ref() };
        Ok(&encoding.ids[..])
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_slice(out, inner) }
}

/// Writes `encoding`'s per-token type ids to `out`, or an empty slice if the tokenizer
/// doesn't produce them. The slice points into `encoding` and is valid until `tk_encoding_free`.
///
/// # Safety
/// 1. `encoding` must be NULL, or a `TkHandle_Encoding` that is not freed while this call runs.
/// 2. `out` must be NULL, or a writable pointer to a `TkSlice_u8`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_type_ids(
    encoding: Handle<Encoding>,
    out: *mut Slice<u8>,
) -> Handle<Error> {
    let inner = move || -> Result<&[u8], &'static str> {
        if encoding.is_null() {
            return Err("encoding must not be NULL");
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let encoding = unsafe { encoding.as_ref() };
        Ok(encoding.type_ids.as_deref().unwrap_or(&[]))
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_slice(out, inner) }
}
