package co.huggingface.tokenizers;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TokenizerFromPretrainedTests {

   @Test
    void testEncodeWordpiece() {
        var tokenizer = TokenizerFromPretrained.create("bert-base-cased").value();
        var input = "Tokenize me please!";
        var encodings = tokenizer.encode(input, true).value();

        var expectedTokens = new String[] {"[CLS]", "To", "##ken", "##ize", "me", "please", "!", "[SEP]"};
        var expectedIds = new long[] {101L, 1706L, 6378L, 3708L, 1143L, 4268L, 106L, 102L};
        var expectedTypeIds = new long[] {0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L};
        var expectedWordIds = new long[] {-1L, 0L, 0L, 0L, 1L, 2L, 3L, -1L};

        var tokens = encodings.getTokens();
        var ids = encodings.getIds();
        var typeIds = encodings.getTypeIds();
        var wordIds = encodings.getWordIds();

        assertArrayEquals(expectedTokens,tokens);
        assertArrayEquals(expectedIds, ids);
        assertArrayEquals(expectedTypeIds, typeIds);
        assertArrayEquals(expectedWordIds, wordIds);

    }

    //cannot fetch model from "google/mt5-base"
    //we can only fetch when there is a tokenizer.json in model hub
    @Test
    void testEncodeUnigram() {
        var tokenizer = TokenizerFromPretrained.create("t5-small").value();
        var input = "Tokenize me please!";
        var encodings = tokenizer.encode(input, true).value();

        var expectedTokens = new String[] {"▁To", "ken", "ize", "▁me", "▁please", "!", "</s>"};
        var expectedIds = new long[] {304L, 2217L, 1737L, 140L, 754L, 55L, 1L};
        var expectedTypeIds = new long[] {0L, 0L, 0L, 0L, 0L, 0L, 0L};
        var expectedWordIds = new long[] {0L, 0L, 0L, 1L, 2L, 2L, -1L};

        var tokens = encodings.getTokens();
        var ids = encodings.getIds();
        var typeIds = encodings.getTypeIds();
        var wordIds = encodings.getWordIds();


        assertArrayEquals(expectedTokens,tokens);
        assertArrayEquals(expectedIds, ids);
        assertArrayEquals(expectedTypeIds, typeIds);
        assertArrayEquals(expectedWordIds, wordIds);

    }

    @Test
    void testEncodeBPE() {
        var tokenizer = TokenizerFromPretrained.create("gpt2").value();
        var input = "Tokenize me please!";
        var encodings = tokenizer.encode(input, true).value();

        var expectedTokens = new String[] {"Token", "ize", "Ġme", "Ġplease", "!"};
        var expectedIds = new long[] {30642L, 1096L, 502L, 3387L, 0L};
        var expectedTypeIds = new long[] {0L, 0L, 0L, 0L, 0L};
        var expectedWordIds = new long[] {0L, 0L, 1L, 2L, 3L};

        var tokens = encodings.getTokens();
        var ids = encodings.getIds();
        var typeIds = encodings.getTypeIds();
        var wordIds = encodings.getWordIds();

        assertArrayEquals(expectedTokens,tokens);
        assertArrayEquals(expectedIds, ids);
        assertArrayEquals(expectedTypeIds, typeIds);
        assertArrayEquals(expectedWordIds, wordIds);

    }

    @Test
    void testWithoutSpecialTokens() {
        var tokenizer = TokenizerFromPretrained.create("bert-base-cased").value();
        var input = "Tokenize me please!";
        var encodings = tokenizer.encode(input, false).value();

        var expectedTokens = new String[] { "To", "##ken", "##ize", "me", "please", "!"};
        var expectedIds = new long[] {1706L, 6378L, 3708L, 1143L, 4268L, 106L};
        var expectedTypeIds = new long[] {0L, 0L, 0L, 0L, 0L, 0L};
        var expectedWordIds = new long[] {0L, 0L, 0L, 1L, 2L, 3L};

        var tokens = encodings.getTokens();
        var ids = encodings.getIds();
        var typeIds = encodings.getTypeIds();
        var wordIds = encodings.getWordIds();

        assertArrayEquals(expectedTokens,tokens);
        assertArrayEquals(expectedIds, ids);
        assertArrayEquals(expectedTypeIds, typeIds);
        assertArrayEquals(expectedWordIds, wordIds);

    }

    @Test
    void testNonExistentModel(){
       String errorMessage = TokenizerFromPretrained.create("boohoo").error();
       assertEquals("Model \"boohoo\" on the Hub doesn't have a tokenizer", errorMessage);
    }

}
