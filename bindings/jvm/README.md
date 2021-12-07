# 🤗 JVM Tokenizers

Provides a JVM binding to the Rust implementation. 

This opens up the use of the tokenizers to various JVM-based languages as Scala, Clojure, Groovy and more, as well as JVM-based frameworks such as SparkNLP and DJL.

The current implementation is a MVP for using 🤗 Tokenizers in JVM production environments.

# Main features
- Exposes `Tokenizer::from_pretrained` to obtain a tokenizer
- Exposes `Tokenizer::encode` call for a single threaded encoding
- Exposes `Tokenizer::encode_batch` call for multi-threaded encoding

# Installation

1. `cd lib` 
2. `./gradlew compileJava`

This will automatically build the Rust FFI-friendly bindings located under `src/main/rust`, as well as the API exposed on the Java side located under `src/main/java`. 

# Benchmarking

Run `./gradlew jmh`

This takes approximately 90 seconds my local and yields the following output:
```
Benchmark                       Mode  Cnt     Score     Error  Units
EncodingBenchmark.singleInput  thrpt    5  1497,624 ± 143,435  ops/s
EncodingBenchmark.batchInput    avgt    2     2,858             s/op
EncodingBenchmark.largeInput    avgt    2     6,452             s/op
```

# Tests

Run `./gradlew test`

Expected output: 
```
> Task :buildRust
    Finished release [optimized] target(s) in 0.45s
    Finished test [unoptimized + debuginfo] target(s) in 0.34s
     Running unittests (src/main/rust/target/debug/deps/safer_ffi_tokenizers-1caa0acafea00370)

running 1 test
test generate_headers ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

# Future Work
- [ ] Support for pair encoding.
- [ ] Support for partially pre-tokenized input.
- [ ] Profile the binding for performance bottlenecks.
- [ ] Add Scala bindings.