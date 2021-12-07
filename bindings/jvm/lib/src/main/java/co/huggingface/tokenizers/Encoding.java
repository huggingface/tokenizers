package co.huggingface.tokenizers;

import co.huggingface.tokenizers.ffi.FFIEncoding;
import com.sun.jna.Pointer;


/**
 * {@code Encoding} wraps a {@link co.huggingface.tokenizers.ffi.FFIEncoding} and defines acessor methods to its fields.
 */
public class Encoding {

    private long[] ids;
    private long[] typeIds;
    private long[] wordIds;
    private String[] tokens;

    /**
     * @param ptr points to an FFIEncoding.
     */
    protected Encoding(Pointer ptr) {
        assert(ptr != null);
        var ffiEncoding = new FFIEncoding(ptr);
        this.ids = ffiEncoding.ids.ptr.getLongArray(0, ffiEncoding.ids.len.intValue());
        this.typeIds = ffiEncoding.type_ids.ptr.getLongArray(0, ffiEncoding.type_ids.len.intValue());
        this.wordIds = ffiEncoding.words.ptr.getLongArray(0, ffiEncoding.words.len.intValue());
        this.tokens = ffiEncoding.tokens.ptr.getStringArray(0, ffiEncoding.tokens.len.intValue());
    }

    /**
     * @return returns tokenized ids
     */
    public long[] getIds() {
        return this.ids;
    }
    /**
     * @return returns the type ids
     */
    public long[] getTypeIds() {
        return this.typeIds;
    }

    /**
     * @return returns the tokenized word indexes
     */
    public long[] getWordIds() {
        return this.wordIds;
    }

    /**
     * @return returns tokenized strings
     */
    public String[] getTokens() {
        return this.tokens;
    }
}
