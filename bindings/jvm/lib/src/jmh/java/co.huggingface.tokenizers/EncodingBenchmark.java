package co.huggingface.tokenizers;

import org.openjdk.jmh.annotations.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


@Threads(value = 1)
@Fork(value = 1)
public class EncodingBenchmark {

    @State(Scope.Benchmark)
    public static class ExecutionPlan {
        private TokenizerFromPretrained tokenizer;
        private String bigInput;
        private List<String> batchInput;
        private String smallInput = "The Project Gutenberg EBook of The Adventures of Sherlock Holmes\n" +
                "by Sir Arthur Conan Doyle\n" +
                "(#15 in our series by Sir Arthur Conan Doyle)\n" +
                "\n" +
                "Copyright laws are changing all over the world. Be sure to check the\n" +
                "copyright laws for your country before downloading or redistributing\n" +
                "this or any other Project Gutenberg eBook.";
        private int batchSize = 20;

        @Setup(Level.Invocation)
        public void setUp() throws IOException {
            bigInput = this.getResourceAsString("big.txt");
            batchInput = this.getBatch(bigInput);
            tokenizer = TokenizerFromPretrained.create("bert-base-cased").value();
        }

        String getResourceAsString(String filename) throws IOException {
            var inputStream = this.getClass().getClassLoader().getResourceAsStream(filename);
            ByteArrayOutputStream result = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024];
            for (int length; (length = inputStream.read(buffer)) != -1; ) {
                result.write(buffer, 0, length);
            }
            return result.toString();
        }

        List<String> getBatch(String input) throws IOException {
            var chunkSize = (input.length() / batchSize);
            var result = new ArrayList<String>(batchSize);
            for(int i = 0; i < batchSize; i++) {
                var substr = input.substring(i * chunkSize, (i + 1) * chunkSize);
                result.add(substr);
            }
            return result;
        }
    }


    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Warmup(iterations = 10, time = 1)
    @Measurement(iterations = 5, time = 1)
    public void singleInput(ExecutionPlan plan) {
        plan.tokenizer.encode(plan.smallInput, false);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @Warmup(iterations = 1)
    @Measurement(iterations = 2)
    public void largeInput(ExecutionPlan plan)  {
        plan.tokenizer.encode(plan.bigInput, false);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @Warmup(iterations = 1)
    @Measurement(iterations = 2)
    public void batchInput(ExecutionPlan plan) {
        plan.tokenizer.encode_batch(plan.batchInput, false);
    }
}