# tokenizers-java

Experimental Java bindings for the tokenizers library, on top of the C ABI in `../c`.

No JNI: one class calls `libtokenizers_c` through the Foreign Function & Memory API, so it
needs **JDK 22+** and no extra native code. Expect breaking changes alongside the C ABI.

## Build & test

```sh
(cd ../c && cargo build --release)
make test
```

## Use

```java
import co.huggingface.tokenizers.Tokenizer;

try (Tokenizer tok = Tokenizer.fromFile(Path.of("tokenizer.json"))) {
    int[] ids = tok.encode("Hello, world!").ids();   // [15496, 11, 995, 0] for gpt2
    String text = tok.decode(ids);
}
```

Run with `--enable-native-access=ALL-UNNAMED -Djava.library.path=<dir with libtokenizers_c>`.
Errors from the library surface as `Tokenizer.TokenizerException`. One `Tokenizer` can be shared
across threads for `encode`/`decode`; don't `close()` it while they run.
