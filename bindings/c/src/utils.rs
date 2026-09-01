use std::{
    ffi::{CStr, c_char},
    str::Utf8Error,
};

use crate::error::{TkError, catch_panic};

/// Marks a type whose instances Rust heap-allocates and frees.
/// Must be wrapped in a [`TkHandle`] to be passed to C code.
pub(crate) trait RustOwned {}

/// An opaque pointer to a Rust-allocated type.
#[repr(transparent)]
pub(crate) struct TkHandle<T>(*mut T);

impl<T> Copy for TkHandle<T> {}

impl<T> Clone for TkHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> TkHandle<T> {
    /// A handle representing "no value", ie a null pointer
    pub(crate) fn null() -> Self {
        TkHandle(std::ptr::null_mut())
    }

    fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Borrows the value the handle points at.
    ///
    /// # Safety
    /// The handle must be non-NULL and point to a valid, not-yet-freed `T`.
    pub(crate) unsafe fn as_ref(&self) -> &T {
        // SAFETY: caller's obligation, documented above.
        unsafe { &*self.0 }
    }
}

/// Leaks `value` onto the heap and returns it as a raw pointer, pairing with [`free_tk_handle`].
pub(crate) fn new_tk_handle<T: RustOwned>(value: T) -> TkHandle<T> {
    TkHandle(Box::into_raw(Box::new(value)))
}

/// Frees `handle` and writes NULL to it, so calling this again on the same pointer is a no-op
/// instead of a double free.
///
/// # Safety
/// `handle` must point to a [`TkHandle<T>`] that is either NULL or a live
/// (not-yet-freed) instance of `T`.
pub(crate) unsafe fn free_tk_handle<T: RustOwned>(handle: *mut TkHandle<T>) {
    // SAFETY: caller's obligation, documented above.
    let handle = unsafe { &mut *handle };
    if handle.is_null() {
        return;
    }
    // SAFETY: just checked non-NULL; liveness is the caller's obligation, documented above.
    drop(unsafe { Box::from_raw(handle.0) });
    *handle = TkHandle::null();
}

/// A wrapper for FFI functions that output a [`TkHandle`].
///
/// Takes care of:
/// - initializing the out pointer to NULL
/// - catching panic unwinds so they don't reach C code
/// - converting the output of `inner` to a [`TkHandle`] when successful
///
/// # Safety
/// `out` must point to valid, writable memory for a [`TkHandle<T>`]. It can be uninitialized.
pub(crate) unsafe fn wrap_in_tk_handle<T: RustOwned, E: std::fmt::Display>(
    out: *mut TkHandle<T>,
    inner: impl FnOnce() -> Result<T, E>,
) -> TkHandle<TkError> {
    // SAFETY: caller's obligation, documented above.
    unsafe { std::ptr::write(out, TkHandle::null()) };
    catch_panic(move || -> Result<(), E> {
        let value = inner()?;
        // SAFETY: caller's obligation, documented above.
        unsafe { std::ptr::write(out, new_tk_handle(value)) };
        Ok(())
    })
}

/// A borrowed view into a Rust-owned slice.
/// Bundles a pointer with the slice length.
/// Valid only as long as whatever it points to is still alive.
#[repr(C)]
pub(crate) struct TkSlice<T> {
    ptr: *const T,
    len: usize,
}

impl<T> TkSlice<T> {
    pub(crate) fn null() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

/// Writes a view of `value` to `out`, bundling pointer and slice length in one [`TkSlice`].
///
/// # Safety
/// `out` must point to valid, writable memory for a [`TkSlice<T>`]. It can be uninitialized.
pub(crate) unsafe fn write_tk_slice<T>(out: *mut TkSlice<T>, value: &[T]) {
    let slice = TkSlice {
        ptr: value.as_ptr(),
        len: value.len(),
    };
    // SAFETY: caller's obligation, documented above.
    unsafe {
        std::ptr::write(out, slice);
    }
}

/// A wrapper for FFI functions that output a [`TkSlice`].
///
/// Takes care of:
/// - initializing the out pointer to an empty slice
/// - catching panic unwinds so they don't reach C code
/// - writing `inner`'s slice to the out pointer, when it succeeds
///
/// A missing value (e.g. an encoding with no type ids) is a `body` that returns `Ok(&[])`: a
/// [`TkSlice`] can't tell "absent" from "empty" apart, and nothing needs it to.
///
/// # Safety
/// `out` must point to valid, writable memory for a [`TkSlice<T>`]. It can be uninitialized.
pub(crate) unsafe fn wrap_in_tk_slice<'a, T: 'a>(
    out: *mut TkSlice<T>,
    inner: impl FnOnce() -> &'a [T],
) -> TkHandle<TkError> {
    // SAFETY: caller's obligation, documented above.
    unsafe { std::ptr::write(out, TkSlice::null()) };
    catch_panic(move || -> Result<(), std::convert::Infallible> {
        let slice = inner();
        // SAFETY: caller's obligation, documented above.
        unsafe { write_tk_slice(out, slice) };
        Ok(())
    })
}

/// Borrows a NUL-terminated C string as a UTF-8 `&str`, without copying.
///
/// # Safety
/// `c_str` must be non-NULL, point to a single NUL-terminated byte string, and be valid for
/// reads up to and including that NUL byte for as long as the returned `&str` is alive. The
/// pointed-to memory must not be mutated during that time.
pub(crate) unsafe fn convert_c_str<'a>(c_str: *const c_char) -> Result<&'a str, Utf8Error> {
    // SAFETY: caller's obligation, documented above.
    let c_str = unsafe { CStr::from_ptr(c_str) };
    c_str.to_str()
}

/// Borrows a `len`-byte buffer as a UTF-8 `&str`, without copying. Unlike [`convert_c_str`],
/// `buf` doesn't need a NUL terminator and may contain embedded NUL bytes.
///
/// # Safety
/// `buf` must be non-NULL and valid for reads of `len` bytes for as long as the returned `&str`
/// is alive. The pointed-to memory must not be mutated during that time.
pub(crate) unsafe fn convert_c_buf<'a>(
    buf: *const c_char,
    len: usize,
) -> Result<&'a str, Utf8Error> {
    // SAFETY: caller's obligation, documented above.
    let bytes = unsafe { std::slice::from_raw_parts(buf as *const u8, len) };
    std::str::from_utf8(bytes)
}
