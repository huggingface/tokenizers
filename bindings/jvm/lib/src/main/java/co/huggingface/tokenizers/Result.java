package co.huggingface.tokenizers;

import co.huggingface.tokenizers.ffi.FFIResult;

/**
 * {@code Result} wraps the result of ffi calls
 */
public class Result<T> {

    private FFIResult ffiResult;

    private T value;

    public Result(FFIResult ffiResult, T value) {
        this.ffiResult = ffiResult;
        this.value = value;
    }

    public T value() {
        return this.value;
    }

    public String error() {
        return ffiResult.error;
    }

    public boolean isError() {
        return ffiResult.error != null;
    }

    public boolean hasValue() {
        return ffiResult.value != null;
    }
}
