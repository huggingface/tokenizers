use crate::error::{Error, catch_panic};
use crate::utils::{RustOwned, free_ptr, wrap_in_ptr};

#[derive(Clone, Copy)]
pub struct EncodeOptions {
    pub(crate) add_special_tokens: bool,
}
impl RustOwned for EncodeOptions {}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
        }
    }
}

/// Creates encode options holding the defaults (`add_special_tokens: true`) and writes their
/// pointer to `out`, or NULL to `out` if this call fails.
///
/// # Safety
/// `out` must be NULL, or a writable pointer to a `TkEncodeOptions *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encode_options_new(out: *mut *mut EncodeOptions) -> *mut Error {
    // SAFETY: caller's obligation, documented above.
    unsafe {
        wrap_in_ptr(out, || -> tk_encode::Result<EncodeOptions> {
            Ok(EncodeOptions::default())
        })
    }
}

/// Frees `options` and writes NULL to it, so calling this again on the same pointer is a no-op
/// instead of a double free.
///
/// # Safety
/// `options` must be NULL, or point to a writable `TkEncodeOptions *` holding NULL or options
/// that are not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encode_options_free(options: *mut *mut EncodeOptions) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_ptr(options) }
}

/// Sets whether encoding adds the tokenizer's configured special tokens (e.g. `[CLS]`/`[SEP]`).
///
/// # Safety
/// `options` must be NULL, or encode options that are not freed and that no other thread is
/// using while this call runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encode_options_set_add_special_tokens(
    options: Option<&mut EncodeOptions>,
    value: bool,
) -> *mut Error {
    catch_panic(move || -> Result<(), &str> {
        let Some(options) = options else {
            return Err("options must not be NULL");
        };
        options.add_special_tokens = value;
        Ok(())
    })
}

#[derive(Clone, Copy)]
pub struct DecodeOptions {
    pub(crate) skip_special_tokens: bool,
}
impl RustOwned for DecodeOptions {}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
        }
    }
}

/// Creates decode options holding the defaults (`skip_special_tokens: true`) and writes their
/// pointer to `out`, or NULL to `out` if this call fails.
///
/// # Safety
/// `out` must be NULL, or a writable pointer to a `TkDecodeOptions *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decode_options_new(out: *mut *mut DecodeOptions) -> *mut Error {
    // SAFETY: caller's obligation, documented above.
    unsafe {
        wrap_in_ptr(out, || -> tk_encode::Result<DecodeOptions> {
            Ok(DecodeOptions::default())
        })
    }
}

/// Frees `options` and writes NULL to it, so calling this again on the same pointer is a no-op
/// instead of a double free.
///
/// # Safety
/// `options` must be NULL, or point to a writable `TkDecodeOptions *` holding NULL or options
/// that are not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decode_options_free(options: *mut *mut DecodeOptions) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_ptr(options) }
}

/// Sets whether decoding omits the tokenizer's configured special tokens from the result.
///
///
/// # Safety
/// `options` must be NULL, or decode options that are not freed and that no other thread is
/// using while this call runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decode_options_set_skip_special_tokens(
    options: Option<&mut DecodeOptions>,
    value: bool,
) -> *mut Error {
    catch_panic(move || -> Result<(), &str> {
        let Some(options) = options else {
            return Err("options must not be NULL");
        };
        options.skip_special_tokens = value;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_encode_options() -> *mut EncodeOptions {
        let mut options = std::ptr::null_mut();
        assert!(unsafe { tk_encode_options_new(&mut options) }.is_null());
        options
    }

    fn new_decode_options() -> *mut DecodeOptions {
        let mut options = std::ptr::null_mut();
        assert!(unsafe { tk_decode_options_new(&mut options) }.is_null());
        options
    }

    #[test]
    fn encode_options_new_holds_the_defaults() {
        let mut options = new_encode_options();
        // SAFETY: options was just created, non-NULL, and nothing else references it.
        assert!(unsafe { &*options }.add_special_tokens);
        unsafe { tk_encode_options_free(&mut options) };
    }

    #[test]
    fn encode_options_new_out_null_reports_an_error_without_creating_one() {
        let err = unsafe { tk_encode_options_new(std::ptr::null_mut()) };
        assert!(!err.is_null());
    }

    #[test]
    fn encode_options_set_changes_the_field() {
        let mut options = new_encode_options();
        // SAFETY: options is live, non-NULL, and nothing else references it.
        assert!(
            unsafe { tk_encode_options_set_add_special_tokens(options.as_mut(), false) }.is_null()
        );
        assert!(!unsafe { &*options }.add_special_tokens);
        unsafe { tk_encode_options_free(&mut options) };
    }

    #[test]
    fn encode_options_set_on_null_reports_an_error() {
        let err = unsafe { tk_encode_options_set_add_special_tokens(None, true) };
        assert!(!err.is_null());
    }

    #[test]
    fn encode_options_free_is_a_no_op_the_second_time_on_the_same_slot() {
        let mut options = new_encode_options();
        unsafe { tk_encode_options_free(&mut options) };
        unsafe { tk_encode_options_free(&mut options) };
        assert!(options.is_null());
    }

    #[test]
    fn decode_options_new_holds_the_defaults() {
        let mut options = new_decode_options();
        // SAFETY: options was just created, non-NULL, and nothing else references it.
        assert!(unsafe { &*options }.skip_special_tokens);
        unsafe { tk_decode_options_free(&mut options) };
    }

    #[test]
    fn decode_options_new_out_null_reports_an_error_without_creating_one() {
        let err = unsafe { tk_decode_options_new(std::ptr::null_mut()) };
        assert!(!err.is_null());
    }

    #[test]
    fn decode_options_set_changes_the_field() {
        let mut options = new_decode_options();
        // SAFETY: options is live, non-NULL, and nothing else references it.
        assert!(
            unsafe { tk_decode_options_set_skip_special_tokens(options.as_mut(), false) }.is_null()
        );
        assert!(!unsafe { &*options }.skip_special_tokens);
        unsafe { tk_decode_options_free(&mut options) };
    }

    #[test]
    fn decode_options_set_on_null_reports_an_error() {
        let err = unsafe { tk_decode_options_set_skip_special_tokens(None, true) };
        assert!(!err.is_null());
    }

    #[test]
    fn decode_options_free_is_a_no_op_the_second_time_on_the_same_slot() {
        let mut options = new_decode_options();
        unsafe { tk_decode_options_free(&mut options) };
        unsafe { tk_decode_options_free(&mut options) };
        assert!(options.is_null());
    }
}
