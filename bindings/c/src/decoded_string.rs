use crate::{
    error::Error,
    utils::{Handle, RustOwned, Slice, free_handle, wrap_in_slice},
};

pub struct DecodedString(pub(crate) String);
impl RustOwned for DecodedString {}

/// Frees `decoded_string` and writes NULL to it, so calling this again on the same pointer is a
/// no-op instead of a double free.
///
/// # Safety
/// `decoded_string` must be non-NULL and point to a `TkHandle_DecodedString` that is either NULL
/// or live (not already freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decoded_string_free(decoded_string: *mut Handle<DecodedString>) {
    // SAFETY: caller's obligation, documented above
    unsafe {
        free_handle(decoded_string);
    }
}

/// Writes `decoded_string`'s UTF-8 bytes to `out`.
///
/// # Safety
/// 1. `decoded_string` must be non-NULL and point to a valid, not-yet-freed
///    `TkHandle_DecodedString`.
/// 2. `out` must be a valid, writable pointer to a `TkSlice_u8`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decoded_string_bytes(
    decoded_string: Handle<DecodedString>,
    out: *mut Slice<u8>,
) -> Handle<Error> {
    let inner = move || -> Result<&[u8], &'static str> {
        if decoded_string.is_null() {
            return Err("decoded_string must not be NULL");
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let decoded_string = unsafe { decoded_string.as_ref() };
        Ok(decoded_string.0.as_bytes())
    };
    // SAFETY: caller's obligation, documented above.
    unsafe { wrap_in_slice(out, inner) }
}
