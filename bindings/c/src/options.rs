use crate::error::{Error, catch_panic};
use crate::utils::{Handle, RustOwned, free_handle, wrap_in_handle};

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
/// handle to `out`, or NULL to `out` if this call fails.
///
/// # Safety
/// `out` must be NULL, or a writable pointer to a `TkHandle_EncodeOptions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encode_options_new(out: *mut Handle<EncodeOptions>) -> Handle<Error> {
    // SAFETY: caller's obligation, documented above.
    unsafe {
        wrap_in_handle(out, || -> tk_encode::Result<EncodeOptions> {
            Ok(EncodeOptions::default())
        })
    }
}

/// Frees `options` and writes NULL to it, so calling this again on the same pointer is safe.
///
/// # Safety
/// `options` must be NULL, or point to a `TkHandle_EncodeOptions` holding NULL or a handle that
/// is not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encode_options_free(options: *mut Handle<EncodeOptions>) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_handle(options) }
}

/// Sets whether encoding adds the tokenizer's configured special tokens (e.g. `[CLS]`/`[SEP]`).
///
/// # Safety
/// `options` must be NULL, or a `TkHandle_EncodeOptions` that is not yet freed and that no other
/// thread is using while this call runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_encode_options_set_add_special_tokens(
    options: Handle<EncodeOptions>,
    value: bool,
) -> Handle<Error> {
    catch_panic(move || -> tk_encode::Result<()> {
        if options.is_null() {
            return Err("options must not be NULL".into());
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let options = unsafe { options.as_mut() };
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
/// handle to `out`, or NULL to `out` if this call fails.
///
/// # Safety
/// `out` must be NULL, or a writable pointer to a `TkHandle_DecodeOptions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decode_options_new(out: *mut Handle<DecodeOptions>) -> Handle<Error> {
    // SAFETY: caller's obligation, documented above.
    unsafe {
        wrap_in_handle(out, || -> tk_encode::Result<DecodeOptions> {
            Ok(DecodeOptions::default())
        })
    }
}

/// Frees `options` and writes NULL to it, so calling this again on the same pointer is safe.
///
/// # Safety
/// `options` must be NULL, or point to a `TkHandle_DecodeOptions` holding NULL or a handle that
/// is not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decode_options_free(options: *mut Handle<DecodeOptions>) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_handle(options) }
}

/// Sets whether decoding omits the tokenizer's configured special tokens from the result.
///
///
/// # Safety
/// `options` must be NULL, or a `TkHandle_DecodeOptions` that is not yet freed and that no other
/// thread is using while this call runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_decode_options_set_skip_special_tokens(
    options: Handle<DecodeOptions>,
    value: bool,
) -> Handle<Error> {
    catch_panic(move || -> tk_encode::Result<()> {
        if options.is_null() {
            return Err("options must not be NULL".into());
        }
        // SAFETY: just checked non-NULL; rest of the caller's obligation is documented above.
        let options = unsafe { options.as_mut() };
        options.skip_special_tokens = value;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_encode_options() -> Handle<EncodeOptions> {
        let mut options = Handle::null();
        assert!(unsafe { tk_encode_options_new(&mut options) }.is_null());
        options
    }

    fn new_decode_options() -> Handle<DecodeOptions> {
        let mut options = Handle::null();
        assert!(unsafe { tk_decode_options_new(&mut options) }.is_null());
        options
    }

    #[test]
    fn encode_options_new_holds_the_defaults() {
        let options = new_encode_options();
        // SAFETY: options was just created, non-NULL, and nothing else references it.
        assert!(unsafe { options.as_ref() }.add_special_tokens);
        let mut options = options;
        unsafe { tk_encode_options_free(&mut options) };
    }

    #[test]
    fn encode_options_new_out_null_reports_an_error_without_creating_one() {
        let err = unsafe { tk_encode_options_new(std::ptr::null_mut()) };
        assert!(!err.is_null());
    }

    #[test]
    fn encode_options_set_changes_the_field() {
        let options = new_encode_options();
        assert!(unsafe { tk_encode_options_set_add_special_tokens(options, false) }.is_null());
        // SAFETY: options is still live, and nothing else references it.
        assert!(!unsafe { options.as_ref() }.add_special_tokens);
        let mut options = options;
        unsafe { tk_encode_options_free(&mut options) };
    }

    #[test]
    fn encode_options_set_on_null_handle_reports_an_error() {
        let err = unsafe { tk_encode_options_set_add_special_tokens(Handle::null(), true) };
        assert!(!err.is_null());
    }

    #[test]
    fn encode_options_free_is_a_no_op_the_second_time_on_the_same_handle() {
        let mut options = new_encode_options();
        unsafe { tk_encode_options_free(&mut options) };
        unsafe { tk_encode_options_free(&mut options) };
        assert!(options.is_null());
    }

    #[test]
    fn decode_options_new_holds_the_defaults() {
        let options = new_decode_options();
        // SAFETY: options was just created, non-NULL, and nothing else references it.
        assert!(unsafe { options.as_ref() }.skip_special_tokens);
        let mut options = options;
        unsafe { tk_decode_options_free(&mut options) };
    }

    #[test]
    fn decode_options_new_out_null_reports_an_error_without_creating_one() {
        let err = unsafe { tk_decode_options_new(std::ptr::null_mut()) };
        assert!(!err.is_null());
    }

    #[test]
    fn decode_options_set_changes_the_field() {
        let options = new_decode_options();
        assert!(unsafe { tk_decode_options_set_skip_special_tokens(options, false) }.is_null());
        // SAFETY: options is still live, and nothing else references it.
        assert!(!unsafe { options.as_ref() }.skip_special_tokens);
        let mut options = options;
        unsafe { tk_decode_options_free(&mut options) };
    }

    #[test]
    fn decode_options_set_on_null_handle_reports_an_error() {
        let err = unsafe { tk_decode_options_set_skip_special_tokens(Handle::null(), true) };
        assert!(!err.is_null());
    }

    #[test]
    fn decode_options_free_is_a_no_op_the_second_time_on_the_same_handle() {
        let mut options = new_decode_options();
        unsafe { tk_decode_options_free(&mut options) };
        unsafe { tk_decode_options_free(&mut options) };
        assert!(options.is_null());
    }
}
