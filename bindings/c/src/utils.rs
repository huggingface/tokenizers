use std::{
    ffi::{CStr, c_char},
    str::Utf8Error,
};

/// Marks a type whose instances Rust heap-allocates and frees.
/// Must be wrapped in a [`TkHandle`] to be passed to C code.
pub(crate) trait RustOwned {}

/// An opaque pointer to a rust-allocated type.
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
    /// The handle must be non-NULL and point to a valid, not-yet-freed [`T`].
    pub(crate) unsafe fn as_ref(&self) -> &T {
        // SAFETY: caller's obligation, documented above.
        unsafe { &*self.0 }
    }
}

/// Leaks `value` onto the heap and returns it as a raw pointer, pairing with [`drop_tk_handle`].
pub(crate) fn make_tk_handle<T: RustOwned>(value: T) -> TkHandle<T> {
    TkHandle(Box::into_raw(Box::new(value)))
}

/// # Safety
/// `handle` must be NULL, or a pointer to a valid (not freed) instance of [`T`].
pub(crate) unsafe fn drop_tk_handle<T: RustOwned>(handle: TkHandle<T>) {
    if handle.is_null() {
        // dropping a NULL pointer is a no-op
        return;
    }
    // SAFETY: caller's obligation, documented above.
    drop(unsafe { Box::from_raw(handle.0) })
}

/// Boxes `value` and writes the resulting handle to `out`, pairing with [`drop_tk_handle`].
///
/// # Safety
/// `out` must point to valid, writable memory for a [`TkHandle<T>`]. It can be uninitialized.
pub(crate) unsafe fn write_tk_handle<T: RustOwned>(out: *mut TkHandle<T>, value: T) {
    unsafe {
        std::ptr::write(out, make_tk_handle(value));
    }
}

/// A borrowed view into a Rust-owned slice.
/// It's a pointer paired with the slice length.
/// Valid only as long as whatever it points to is still alive.
#[repr(C)]
pub(crate) struct TkSlice<T> {
    ptr: *const T,
    len: usize,
}

/// Writes a view of `value` to `out`, pairing pointer and length in one [`TkSlice`].
///
/// # Safety
/// `out` must point to valid, writable memory for a [`TkSlice<T>`]. It can be uninitialized.
pub(crate) unsafe fn write_c_slice<T>(out: *mut TkSlice<T>, value: &[T]) {
    let slice = TkSlice {
        ptr: value.as_ptr(),
        len: value.len(),
    };
    // SAFETY: caller's obligation, documented above.
    unsafe {
        std::ptr::write(out, slice);
    }
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
