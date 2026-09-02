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
/// `decoded_string` must be NULL, or point to a `TkHandle_DecodedString` holding NULL or a
/// handle that is not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decoded_string_free(decoded_string: *mut Handle<DecodedString>) {
    // SAFETY: caller's obligation, documented above.
    unsafe {
        free_handle(decoded_string);
    }
}

/// Writes `decoded_string`'s UTF-8 bytes to `out`. The slice points into `decoded_string` and is
/// valid until `tk_decoded_string_free`.
///
/// # Safety
/// 1. `decoded_string` must be NULL, or a `TkHandle_DecodedString` that is not freed while this
///    call runs.
/// 2. `out` must be NULL, or a writable pointer to a `TkSlice_u8`.
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
