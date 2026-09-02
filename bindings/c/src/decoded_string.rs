use crate::{
    error::Error,
    utils::{RustOwned, Slice, free_ptr, wrap_in_slice},
};

pub struct DecodedString(pub(crate) String);
impl RustOwned for DecodedString {}

/// Frees `decoded_string` and writes NULL to it, so calling this again on the same pointer is a
/// no-op instead of a double free.
///
/// # Safety
/// `decoded_string` must be NULL, or point to a writable `TkDecodedString *` holding NULL or a
/// string that is not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decoded_string_free(decoded_string: *mut *mut DecodedString) {
    // SAFETY: caller's obligation, documented above.
    unsafe {
        free_ptr(decoded_string);
    }
}

/// Writes `decoded_string`'s UTF-8 bytes to `out`. The slice points into `decoded_string` and is
/// valid until `tk_decoded_string_free`.
///
/// # Safety
/// 1. `decoded_string` must be NULL, or a string that is not freed while this call runs.
/// 2. `out` must be NULL, or a writable pointer to a `TkSlice_u8`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decoded_string_bytes(
    decoded_string: Option<&DecodedString>,
    out: *mut Slice<u8>,
) -> *mut Error {
    let inner = move || -> Result<&[u8], &'static str> {
        let Some(decoded_string) = decoded_string else {
            return Err("decoded_string must not be NULL");
        };
        Ok(decoded_string.0.as_bytes())
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_slice(out, inner) }
}
