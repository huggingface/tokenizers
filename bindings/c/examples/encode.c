#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizers/tokenizers.h"

static void handle_error(TkHandle_TkError err)
{
    if (err != NULL)
    {
        (void)fprintf(stderr, "tokenizers-c error: %s\n", tk_error_message(err));
        tk_error_free(&err);
        exit(1);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        (void)fprintf(stderr, "usage: %s <tokenizer.json> [text]\n", argv[0]);
        return 1;
    }
    const char *tokenizer_path = argv[1];
    const char *text = argc > 2 ? argv[2] : "Hello, world!";

    TkHandle_TkTokenizer tokenizer = NULL;
    handle_error(tk_tokenizer_from_file(tokenizer_path, &tokenizer));
    printf("loaded tokenizer from %s\n", tokenizer_path);

    TkHandle_TkEncoding encoding = NULL;
    handle_error(tk_tokenizer_encode(tokenizer, text, strlen(text), true, &encoding));

    struct TkSlice_u32 ids;
    handle_error(tk_encoding_ids(encoding, &ids));
    printf("encoded \"%s\" into %zu token(s):", text, ids.len);
    for (size_t i = 0; i < ids.len; i++)
    {
        printf(" %u", ids.ptr[i]);
    }
    printf("\n");

    struct TkSlice_u8 type_ids;
    handle_error(tk_encoding_type_ids(encoding, &type_ids));
    if (type_ids.len > 0)
    {
        printf("type ids:");
        for (size_t i = 0; i < type_ids.len; i++)
        {
            printf(" %u", type_ids.ptr[i]);
        }
        printf("\n");
    }

    tk_encoding_free(&encoding);
    tk_tokenizer_free(&tokenizer);
    return 0;
}
