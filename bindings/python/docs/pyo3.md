# PyO3 Usage Notes

## Why we take `self_: PyRef<'_, Self>`

Most of the Python-facing structs are declared with `#[pyclass(extends = ...)]`. The actual data (for example the `processor` field in `PyPostProcessor`) lives in the base class, while the derived Rust structs are often just markers so that Python sees a proper subclass. When we implement a method on the subclass, we still need to reach into the base storage without downcasting a `PyAny` or re-wrapping objects.

Using `self_: PyRef<'_, Self>` gives us a borrowed reference to the Python-owned value that keeps the GIL lifetime, reference counts, and the inheritance chain intact. With it we can call `self_.as_ref()` to view the base `PyPostProcessor` directly and access shared helpers like the processor getters/setters. If we used a plain `&self` we would only see the zero-sized derived struct and would have to convert through a super type just to touch the processors, which adds boilerplate and loses the link to the Python inheritance model. This is the PyO3 equivalent of Python’s `super()`—it keeps the Rust type information while letting us operate on the underlying parent.
