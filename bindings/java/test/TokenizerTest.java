import co.huggingface.tokenizers.Tokenizer;
import java.nio.file.Path;
import java.util.Arrays;

/** Run with -ea. Exits non-zero on the first failed assertion. */
public class TokenizerTest {
    public static void main(String[] args) {
        Path gpt2 = Path.of(args[0]);

        try (Tokenizer tok = Tokenizer.fromFile(gpt2)) {
            int[] ids = tok.encode("Hello, world!").ids();
            assert Arrays.equals(ids, new int[] {15496, 11, 995, 0}) : Arrays.toString(ids);
            assert tok.decode(ids).equals("Hello, world!");
            assert tok.decode(ids, false).equals("Hello, world!");
            assert tok.encode("", false).ids().length == 0;
            assert tok.decode(new int[0]).isEmpty();

            String multi = "héllo 世界 🤗";
            assert tok.decode(tok.encode(multi).ids()).equals(multi);
        }

        try {
            Tokenizer.fromFile(Path.of("does-not-exist.json"));
            throw new AssertionError("expected TokenizerException");
        } catch (Tokenizer.TokenizerException e) {
            assert !e.getMessage().isEmpty();
        }

        Tokenizer closed = Tokenizer.fromFile(gpt2);
        closed.close();
        closed.close(); // idempotent
        try {
            closed.encode("x");
            throw new AssertionError("expected IllegalStateException");
        } catch (IllegalStateException expected) {
        }

        System.out.println("ok");
    }
}
