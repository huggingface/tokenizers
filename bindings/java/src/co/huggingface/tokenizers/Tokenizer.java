package co.huggingface.tokenizers;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/**
 * A tokenizer loaded from a {@code tokenizer.json}, backed by {@code libtokenizers_c} through the
 * Foreign Function & Memory API (JDK 22+). No JNI glue: every method is a direct downcall into
 * {@code bindings/c/include/tokenizers/tokenizers.h}, whose conventions (errors, ownership, threads)
 * apply here unchanged.
 *
 * <p>Thread safety: {@link #encode} and {@link #decode} may be called concurrently on one instance.
 * {@link #close} must not race with them.
 */
public final class Tokenizer implements AutoCloseable {

    /** Token ids and per-token type ids (empty if the tokenizer doesn't produce them). */
    public record Encoding(int[] ids, byte[] typeIds) {}

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIB;

    static {
        // Resolved through -Djava.library.path, like any JNI library.
        System.loadLibrary("tokenizers_c");
        LIB = SymbolLookup.loaderLookup();
    }

    // ponytail: size_t as JAVA_LONG, i.e. 64-bit targets only.
    private static final StructLayout SLICE = MemoryLayout.structLayout(ADDRESS, JAVA_LONG);

    private static MethodHandle fn(String name, MemoryLayout ret, MemoryLayout... args) {
        FunctionDescriptor d = ret == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(ret, args);
        return LINKER.downcallHandle(LIB.find(name).orElseThrow(), d);
    }

    private static final MethodHandle ERROR_MESSAGE = fn("tk_error_message", ADDRESS, ADDRESS);
    private static final MethodHandle ERROR_FREE = fn("tk_error_free", null, ADDRESS);
    private static final MethodHandle FROM_FILE = fn("tk_tokenizer_from_file", ADDRESS, ADDRESS, ADDRESS);
    private static final MethodHandle TOKENIZER_FREE = fn("tk_tokenizer_free", null, ADDRESS);
    private static final MethodHandle ENCODE = fn("tk_tokenizer_encode", ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS);
    private static final MethodHandle ENCODING_IDS = fn("tk_encoding_ids", SLICE, ADDRESS);
    private static final MethodHandle ENCODING_TYPE_IDS = fn("tk_encoding_type_ids", SLICE, ADDRESS);
    private static final MethodHandle ENCODING_FREE = fn("tk_encoding_free", null, ADDRESS);
    private static final MethodHandle ENCODE_OPTIONS_NEW = fn("tk_encode_options_new", ADDRESS, ADDRESS);
    private static final MethodHandle ENCODE_OPTIONS_SET_ADD_SPECIAL = fn("tk_encode_options_set_add_special_tokens", ADDRESS, ADDRESS, JAVA_BOOLEAN);
    private static final MethodHandle ENCODE_OPTIONS_FREE = fn("tk_encode_options_free", null, ADDRESS);
    private static final MethodHandle DECODE = fn("tk_tokenizer_decode", ADDRESS, ADDRESS, SLICE, ADDRESS, ADDRESS);
    private static final MethodHandle DECODE_OPTIONS_NEW = fn("tk_decode_options_new", ADDRESS, ADDRESS);
    private static final MethodHandle DECODE_OPTIONS_SET_SKIP_SPECIAL = fn("tk_decode_options_set_skip_special_tokens", ADDRESS, ADDRESS, JAVA_BOOLEAN);
    private static final MethodHandle DECODE_OPTIONS_FREE = fn("tk_decode_options_free", null, ADDRESS);
    private static final MethodHandle STRING_CSTR = fn("tk_string_cstr", ADDRESS, ADDRESS);
    private static final MethodHandle STRING_LEN = fn("tk_string_len", JAVA_LONG, ADDRESS);
    private static final MethodHandle STRING_FREE = fn("tk_string_free", null, ADDRESS);

    private MemorySegment handle;

    private Tokenizer(MemorySegment handle) {
        this.handle = handle;
    }

    /** Loads a tokenizer from a {@code tokenizer.json} file. */
    public static Tokenizer fromFile(Path path) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            check(arena, (MemorySegment) FROM_FILE.invokeExact(arena.allocateFrom(path.toString()), out));
            return new Tokenizer(out.get(ADDRESS, 0));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Encodes {@code text}, adding the tokenizer's special tokens. */
    public Encoding encode(String text) {
        return encode(text, true);
    }

    public Encoding encode(String text, boolean addSpecialTokens) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment options = MemorySegment.NULL; // NULL = defaults (add_special_tokens: true)
            if (!addSpecialTokens) {
                options = newOptions(arena, ENCODE_OPTIONS_NEW, ENCODE_OPTIONS_SET_ADD_SPECIAL, ENCODE_OPTIONS_FREE, false);
            }
            byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
            MemorySegment out = arena.allocate(ADDRESS);
            try {
                check(arena, (MemorySegment) ENCODE.invokeExact(handle(), arena.allocateFrom(JAVA_BYTE, utf8), (long) utf8.length, options, out));
            } finally {
                free(arena, ENCODE_OPTIONS_FREE, options);
            }
            MemorySegment encoding = out.get(ADDRESS, 0);
            try {
                MemorySegment ids = (MemorySegment) ENCODING_IDS.invokeExact((SegmentAllocator) arena, encoding);
                MemorySegment typeIds = (MemorySegment) ENCODING_TYPE_IDS.invokeExact((SegmentAllocator) arena, encoding);
                return new Encoding(slice(ids, 4).toArray(JAVA_INT), slice(typeIds, 1).toArray(JAVA_BYTE));
            } finally {
                free(arena, ENCODING_FREE, encoding);
            }
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Decodes {@code ids} back into text, skipping special tokens. */
    public String decode(int[] ids) {
        return decode(ids, true);
    }

    public String decode(int[] ids, boolean skipSpecialTokens) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment options = MemorySegment.NULL; // NULL = defaults (skip_special_tokens: true)
            if (!skipSpecialTokens) {
                options = newOptions(arena, DECODE_OPTIONS_NEW, DECODE_OPTIONS_SET_SKIP_SPECIAL, DECODE_OPTIONS_FREE, false);
            }
            MemorySegment slice = arena.allocate(SLICE);
            slice.set(ADDRESS, 0, ids.length == 0 ? MemorySegment.NULL : arena.allocateFrom(JAVA_INT, ids));
            slice.set(JAVA_LONG, ADDRESS.byteSize(), ids.length);
            MemorySegment out = arena.allocate(ADDRESS);
            try {
                check(arena, (MemorySegment) DECODE.invokeExact(handle(), slice, options, out));
            } finally {
                free(arena, DECODE_OPTIONS_FREE, options);
            }
            MemorySegment string = out.get(ADDRESS, 0);
            try {
                long len = (long) STRING_LEN.invokeExact(string);
                MemorySegment cstr = (MemorySegment) STRING_CSTR.invokeExact(string);
                return new String(cstr.reinterpret(len).toArray(JAVA_BYTE), StandardCharsets.UTF_8);
            } finally {
                free(arena, STRING_FREE, string);
            }
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    @Override
    public void close() {
        if (handle != null) {
            try (Arena arena = Arena.ofConfined()) {
                free(arena, TOKENIZER_FREE, handle);
            }
            handle = null;
        }
    }

    private MemorySegment handle() {
        if (handle == null) throw new IllegalStateException("tokenizer is closed");
        return handle;
    }

    private static MemorySegment newOptions(Arena arena, MethodHandle create, MethodHandle set, MethodHandle freeFn, boolean value) throws Throwable {
        MemorySegment out = arena.allocate(ADDRESS);
        check(arena, (MemorySegment) create.invokeExact(out));
        MemorySegment options = out.get(ADDRESS, 0);
        try {
            check(arena, (MemorySegment) set.invokeExact(options, value));
        } catch (Throwable t) {
            free(arena, freeFn, options);
            throw t;
        }
        return options;
    }

    /** Views a by-value {@code TkSlice*} as a segment of {@code len * elemSize} bytes. */
    private static MemorySegment slice(MemorySegment s, long elemSize) {
        long len = s.get(JAVA_LONG, ADDRESS.byteSize());
        return len == 0 ? MemorySegment.NULL : s.get(ADDRESS, 0).reinterpret(len * elemSize);
    }

    /** Every {@code tk_*_free} takes a pointer to the pointer, and NULL is a no-op. */
    private static void free(Arena arena, MethodHandle freeFn, MemorySegment ptr) {
        try {
            freeFn.invokeExact(arena.allocateFrom(ADDRESS, ptr));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Throws the {@code TkError} as a {@link TokenizerException} (and frees it) if non-NULL. */
    private static void check(Arena arena, MemorySegment err) throws Throwable {
        if (err.equals(MemorySegment.NULL)) return;
        MemorySegment msg = (MemorySegment) ERROR_MESSAGE.invokeExact(err);
        String message = msg.reinterpret(Long.MAX_VALUE).getString(0);
        free(arena, ERROR_FREE, err);
        throw new TokenizerException(message);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new RuntimeException(t);
    }

    /** An error reported by {@code libtokenizers_c}. */
    public static final class TokenizerException extends RuntimeException {
        TokenizerException(String message) {
            super(message);
        }
    }
}
