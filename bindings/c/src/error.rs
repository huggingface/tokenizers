use crate::utils::{RustOwned, TkHandle, free_tk_handle, new_tk_handle};
use std::ffi::{CString, c_char};
use std::panic::{self, AssertUnwindSafe};

pub struct TkError {
    message: CString,
}

impl RustOwned for TkError {}

impl TkError {
    pub(crate) fn into_handle(msg: impl std::fmt::Display) -> TkHandle<TkError> {
        let message = CString::new(msg.to_string())
            .unwrap_or_else(|_| CString::new("error message contained a NUL byte").unwrap());
        new_tk_handle(TkError { message })
    }
}

/// Returns a pointer to `err`'s message, or NULL if `err` is NULL. The pointer is owned by `err`
/// and only valid until `err` is freed with `tk_error_free`.
///
/// # Safety
/// `err` must be NULL, or a live pointer returned by a fallible FFI fn and not yet freed with
/// `tk_error_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_error_message(err: *const TkError) -> *const c_char {
    if err.is_null() {
        return std::ptr::null();
    }
    unsafe { &*err }.message.as_ptr()
}

/// Frees `err` and writes NULL to it, so calling this again on the same pointer is a no-op
/// instead of a double free.
///
/// # Safety
/// `err` must be non-NULL and point to a [`TkHandle<TkError>`] that is either NULL or a live
/// (not-yet-freed) error returned by a fallible FFI fn.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_error_free(err: *mut TkHandle<TkError>) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_tk_handle(err) }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Catches unwind panic and converts it to an error
pub(crate) fn catch_panic<E: std::fmt::Display>(
    body: impl FnOnce() -> Result<(), E>,
) -> TkHandle<TkError> {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => TkHandle::null(),
        Ok(Err(err)) => TkError::into_handle(err),
        Err(payload) => TkError::into_handle(panic_message(payload)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message_of(handle: TkHandle<TkError>) -> String {
        assert!(!handle.is_null());
        let err = unsafe { handle.as_ref() };
        err.message.to_str().unwrap().to_string()
    }

    #[test]
    fn ok_body_returns_a_null_handle() {
        let handle = catch_panic(|| -> Result<(), &str> { Ok(()) });
        assert!(handle.is_null());
    }

    #[test]
    fn err_body_carries_its_display_message() {
        let handle = catch_panic(|| -> Result<(), &str> { Err("bad input") });
        assert_eq!(message_of(handle), "bad input");
    }

    #[test]
    fn a_str_panic_payload_is_preserved() {
        // `panic!("literal")` with no format args panics with a `&'static str` payload.
        let handle = catch_panic(|| -> Result<(), &str> { panic!("kaboom") });
        assert_eq!(message_of(handle), "kaboom");
    }

    #[test]
    fn a_string_panic_payload_is_preserved() {
        // Any format args route the panic payload through `format!`, producing a `String`
        // instead of a `&str`.
        let handle = catch_panic(|| -> Result<(), &str> { panic!("{}", "kaboom") });
        assert_eq!(message_of(handle), "kaboom");
    }

    #[test]
    fn an_unrecognized_panic_payload_falls_back_to_a_generic_message() {
        let handle = catch_panic(|| -> Result<(), &str> {
            panic::panic_any(42_i32);
        });
        assert_eq!(message_of(handle), "unknown panic");
    }
}
