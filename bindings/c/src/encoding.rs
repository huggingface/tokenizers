use crate::utils::{Handle, RustOwned, Slice, free_handle, wrap_in_slice};
use crate::error::Error;

pub struct Encoding {
    pub(crate) ids: Vec<u32>,
    pub(crate) type_ids: Option<Vec<u8>>,
}
impl RustOwned for Encoding {}

/// Frees `encoding` and writes NULL to it, so calling this again on the same pointer is a
/// no-op instead of a double free.
///
/// # Safety
/// `encoding` must be non-NULL and point to a `TkHandle_Encoding` that is either NULL
/// or live (not already freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encoding_free(encoding: *mut Handle<Encoding>) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_handle(encoding) }
}

/// Writes `encoding`'s token ids to `out`.
///
/// # Safety
/// 1. `encoding` must be non-NULL and point to a valid, not-yet-freed `TkHandle_Encoding`.
/// 2. `out` must be a valid, writable pointer to a `TkSlice_u32`.
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
/// doesn't produce them.
///
/// # Safety
/// 1. `encoding` must be non-NULL and point to a valid, not-yet-freed `TkHandle_Encoding`.
/// 2. `out` must be a valid, writable pointer to a `TkSlice_u8`.
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
