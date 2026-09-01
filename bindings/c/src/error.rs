use crate::utils::{RustOwned, TkHandle, drop_tk_handle, make_tk_handle};
use std::ffi::{CString, c_char};
use std::panic::{self, AssertUnwindSafe};

pub struct TkError {
    message: CString,
}

impl RustOwned for TkError {}

impl TkError {
    fn into_raw(msg: impl std::fmt::Display) -> TkHandle<TkError> {
        let message = CString::new(msg.to_string())
            .unwrap_or_else(|_| CString::new("error message contained a NUL byte").unwrap());
        make_tk_handle(TkError { message })
    }
}

/// Returns a pointer to `err`'s message. The pointer is owned by `err` and only valid until `err` is
/// freed with `tk_error_free`.
///
/// # Safety
/// `err` must be a live pointer returned by a fallible FFI fn, not yet freed with
/// `tk_error_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_error_message(err: *const TkError) -> *const c_char {
    unsafe { &*err }.message.as_ptr()
}

/// Frees an error returned by a fallible FFI fn. Calling this on a NULL pointer is a no-op.
///
/// # Safety
/// `err` must either be NULL ora live pointer returned by a fallible FFI fn, not yet freed with
/// `tk_error_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_error_free(err: TkHandle<TkError>) {
    unsafe { drop_tk_handle(err) }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

pub(crate) fn catch_panic<E: std::fmt::Display>(
    body: impl FnOnce() -> Result<(), E>,
) -> TkHandle<TkError> {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => TkHandle::null(),
        Ok(Err(err)) => TkError::into_raw(err),
        Err(payload) => TkError::into_raw(panic_message(payload)),
    }
}
