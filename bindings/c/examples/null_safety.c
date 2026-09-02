// Every fallible tk_* function must report a clean TkError instead of crashing when a
// required pointer argument is NULL. Every tk_*_free function must be a safe no-op on NULL,
// and safe to call twice on the same handle. This exercises every such case against the real
// dylib, the same way quick_start.c exercises the happy path.
//
// It also exercises TkEncodeOptions/TkDecodeOptions: NULL means defaults, a set option carries
// through to the encode/decode call, and every set rejects a NULL options handle the same way
// every other tk_* function rejects a NULL handle.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizers/tokenizers.h"

static int failures = 0;

static void check(int ok, const char *what)
{
    if (ok)
    {
        printf("ok: %s\n", what);
    }
    else
    {
        (void)fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static TkHandle_Tokenizer load_tokenizer(const char *path)
{
    TkHandle_Tokenizer tok = NULL;
    TkHandle_Error err = tk_tokenizer_from_file(path, &tok);
    if (err != NULL)
    {
        (void)fprintf(stderr, "setup: tk_tokenizer_from_file failed: %s\n", tk_error_message(err));
        exit(2);
    }
    return tok;
}

static TkHandle_Encoding encode_hello(TkHandle_Tokenizer tok)
{
    TkHandle_Encoding enc = NULL;
    const char *text = "Hello, world!";
    TkHandle_Error err = tk_tokenizer_encode(tok, text, strlen(text), NULL, &enc);
    if (err != NULL)
    {
        (void)fprintf(stderr, "setup: tk_tokenizer_encode failed: %s\n", tk_error_message(err));
        exit(2);
    }
    return enc;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        (void)fprintf(stderr, "usage: %s <tokenizer.json>\n", argv[0]);
        return 1;
    }
    const char *fixture = argv[1];

    check(tk_error_message(NULL) == NULL, "tk_error_message(NULL) returns NULL");

    tk_tokenizer_free(NULL);
    tk_encoding_free(NULL);
    tk_decoded_string_free(NULL);
    tk_error_free(NULL);
    tk_encode_options_free(NULL);
    tk_decode_options_free(NULL);
    printf("ok: tk_*_free(NULL) doesn't crash\n");

    {
        TkHandle_Tokenizer tok = load_tokenizer(fixture);
        tk_tokenizer_free(&tok);
        tk_tokenizer_free(&tok);
        check(tok == NULL, "tk_tokenizer_free() is a no-op the second time on the same handle");
    }

    {
        TkHandle_Tokenizer tok = NULL;
        TkHandle_Error err = tk_tokenizer_from_file(NULL, &tok);
        check(err != NULL && tok == NULL, "tk_tokenizer_from_file(NULL path) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_Error err = tk_tokenizer_from_file(fixture, NULL);
        check(err != NULL, "tk_tokenizer_from_file(..., NULL out) reports an error");
        tk_error_free(&err);
    }

    TkHandle_Tokenizer tok = load_tokenizer(fixture);

    {
        TkHandle_Encoding enc = NULL;
        TkHandle_Error err = tk_tokenizer_encode(NULL, "hi", 2, NULL, &enc);
        check(err != NULL && enc == NULL, "tk_tokenizer_encode(NULL tokenizer) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_Encoding enc = NULL;
        TkHandle_Error err = tk_tokenizer_encode(tok, NULL, 5, NULL, &enc);
        check(err != NULL && enc == NULL,
              "tk_tokenizer_encode(NULL input, len=5) reports an error");
        tk_error_free(&err);
    }
    {
        // NULL input is rejected even at len=0: a TkSlice can't tell "no text" apart from
        // an actually-invalid pointer, so it's simplest and safest to just require non-NULL.
        TkHandle_Encoding enc = NULL;
        TkHandle_Error err = tk_tokenizer_encode(tok, NULL, 0, NULL, &enc);
        check(err != NULL && enc == NULL,
              "tk_tokenizer_encode(NULL input, len=0) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_Error err = tk_tokenizer_encode(tok, "hi", 2, NULL, NULL);
        check(err != NULL, "tk_tokenizer_encode(..., NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_EncodeOptions options = NULL;
        TkHandle_Error err = tk_encode_options_set_add_special_tokens(options, true);
        check(err != NULL, "tk_encode_options_set_add_special_tokens(NULL options) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_Error err = tk_encode_options_new(NULL);
        check(err != NULL, "tk_encode_options_new(NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_EncodeOptions options = NULL;
        TkHandle_Error err = tk_encode_options_new(&options);
        check(err == NULL && options != NULL, "tk_encode_options_new() succeeds");
        err = tk_encode_options_set_add_special_tokens(options, false);
        check(err == NULL, "tk_encode_options_set_add_special_tokens(options, false) succeeds");

        TkHandle_Encoding enc = NULL;
        err = tk_tokenizer_encode(tok, "hi", 2, options, &enc);
        check(err == NULL && enc != NULL, "tk_tokenizer_encode(explicit options) succeeds");
        tk_encoding_free(&enc);

        tk_encode_options_free(&options);
        tk_encode_options_free(&options);
        check(options == NULL, "tk_encode_options_free() is a no-op the second time on the same handle");
    }

    TkHandle_Encoding enc = encode_hello(tok);

    {
        struct TkSlice_u32 out;
        TkHandle_Error err = tk_encoding_ids(NULL, &out);
        check(err != NULL, "tk_encoding_ids(NULL encoding) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_Error err = tk_encoding_ids(enc, NULL);
        check(err != NULL, "tk_encoding_ids(..., NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        struct TkSlice_u8 out;
        TkHandle_Error err = tk_encoding_type_ids(NULL, &out);
        check(err != NULL, "tk_encoding_type_ids(NULL encoding) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_Error err = tk_encoding_type_ids(enc, NULL);
        check(err != NULL, "tk_encoding_type_ids(..., NULL out) reports an error");
        tk_error_free(&err);
    }

    {
        struct TkSlice_u32 ids = {0};
        TkHandle_DecodedString dec = NULL;
        TkHandle_Error err = tk_tokenizer_decode(NULL, ids, NULL, &dec);
        check(err != NULL && dec == NULL, "tk_tokenizer_decode(NULL tokenizer) reports an error");
        tk_error_free(&err);
    }
    {
        struct TkSlice_u32 ids = {0};
        TkHandle_Error err = tk_tokenizer_decode(tok, ids, NULL, NULL);
        check(err != NULL, "tk_tokenizer_decode(..., NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        // Unlike tk_tokenizer_encode's input, a NULL ids.ptr is valid as long as len is 0: a
        // TkSlice can't tell "no ids" apart from an actually-invalid pointer, and here "no ids"
        // is a legitimate, well-defined input (decodes to an empty string).
        struct TkSlice_u32 ids = {NULL, 0};
        TkHandle_DecodedString dec = NULL;
        TkHandle_Error err = tk_tokenizer_decode(tok, ids, NULL, &dec);
        check(err == NULL && dec != NULL,
              "tk_tokenizer_decode(NULL ids.ptr, len=0) succeeds as an empty decode");
        tk_decoded_string_free(&dec);
    }
    {
        TkHandle_DecodeOptions options = NULL;
        TkHandle_Error err = tk_decode_options_set_skip_special_tokens(options, true);
        check(err != NULL, "tk_decode_options_set_skip_special_tokens(NULL options) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_Error err = tk_decode_options_new(NULL);
        check(err != NULL, "tk_decode_options_new(NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_DecodeOptions options = NULL;
        TkHandle_Error err = tk_decode_options_new(&options);
        check(err == NULL && options != NULL, "tk_decode_options_new() succeeds");
        err = tk_decode_options_set_skip_special_tokens(options, false);
        check(err == NULL, "tk_decode_options_set_skip_special_tokens(options, false) succeeds");

        struct TkSlice_u32 ids = {NULL, 0};
        TkHandle_DecodedString dec = NULL;
        err = tk_tokenizer_decode(tok, ids, options, &dec);
        check(err == NULL && dec != NULL, "tk_tokenizer_decode(explicit options) succeeds");
        tk_decoded_string_free(&dec);

        tk_decode_options_free(&options);
        tk_decode_options_free(&options);
        check(options == NULL, "tk_decode_options_free() is a no-op the second time on the same handle");
    }

    {
        struct TkSlice_u32 ids = {NULL, 0};
        TkHandle_DecodedString dec = NULL;
        TkHandle_Error err = tk_tokenizer_decode(tok, ids, NULL, &dec);
        if (err != NULL)
        {
            (void)fprintf(stderr, "setup: tk_tokenizer_decode failed: %s\n", tk_error_message(err));
            exit(2);
        }

        {
            struct TkSlice_u8 out;
            TkHandle_Error bytes_err = tk_decoded_string_bytes(NULL, &out);
            check(bytes_err != NULL, "tk_decoded_string_bytes(NULL decoded_string) reports an error");
            tk_error_free(&bytes_err);
        }
        {
            TkHandle_Error bytes_err = tk_decoded_string_bytes(dec, NULL);
            check(bytes_err != NULL, "tk_decoded_string_bytes(..., NULL out) reports an error");
            tk_error_free(&bytes_err);
        }

        tk_decoded_string_free(&dec);
        tk_decoded_string_free(&dec);
        check(dec == NULL, "tk_decoded_string_free() is a no-op the second time on the same handle");
    }

    tk_encoding_free(&enc);
    tk_tokenizer_free(&tok);

    if (failures > 0)
    {
        (void)fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all null-safety checks passed\n");
    return 0;
}
