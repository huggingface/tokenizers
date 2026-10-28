# Added-token matcher, job 6ae20e173fb22a235dd41771

- machine AMD EPYC 7R13 Processor, nproc 32, kernel 6.12.113-138.209.amzn2023.x86_64, pinned to taskset -c 0
- branch: main + branch.diff (142 added lines): perf/aho-corasick @ d9a5e8af+dirty
- branch Cargo.lock: version = "1.1.5" version = "2.8.3" 
- 15:18:14 end, steal ticks 47

Throughput is input MB/s, one thread, the best of the rounds (n = rounds per arm). Ratio = aho-corasick / buckets: above 1 means aho-corasick is faster.
Spread = (max - min) / max over the rounds, worst arm.

Identical outputs between arms: all cells

## Matcher alone: every match, restarting after each, like `extract_next`

| model | corpus | calls | matches | buckets | aho-corasick | ratio | spread | n |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | agentic | 4k | 32929 | 5892 | 5304 | 0.90 | 4% | 5 |
| DeepSeek-V3 | agentic | whole | 32929 | 6203 | 5568 | 0.90 | 4% | 5 |
| DeepSeek-V3 | code | 4k | 0 | 25563 | 26263 | 1.03 | 3% | 5 |
| DeepSeek-V3 | code | whole | 0 | 32507 | 30898 | 0.95 | 2% | 5 |
| DeepSeek-V3 | english | 4k | 0 | 48469 | 53942 | 1.11 | 5% | 5 |
| DeepSeek-V3 | english | whole | 0 | 71565 | 70561 | 0.99 | 5% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | agentic | 4k | 65998 | 1408 | 3439 | 2.44 | 2% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | agentic | whole | 65998 | 1415 | 3583 | 2.53 | 2% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | code | 4k | 1 | 1652 | 8576 | 5.19 | 1% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | code | whole | 1 | 1649 | 8898 | 5.39 | 2% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | english | 4k | 0 | 1737 | 43477 | 25.03 | 5% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | english | whole | 0 | 1771 | 54415 | 30.73 | 6% | 5 |
| Qwen3-8B | agentic | 4k | 53562 | 4945 | 6523 | 1.32 | 2% | 5 |
| Qwen3-8B | agentic | whole | 53562 | 5156 | 6765 | 1.31 | 2% | 5 |
| Qwen3-8B | code | 4k | 0 | 24868 | 26972 | 1.08 | 4% | 5 |
| Qwen3-8B | code | whole | 0 | 31189 | 32025 | 1.03 | 3% | 5 |
| Qwen3-8B | english | 4k | 0 | 48378 | 55430 | 1.15 | 4% | 5 |
| Qwen3-8B | english | whole | 0 | 70317 | 71433 | 1.02 | 6% | 5 |
| gemma-3-27b-it-int4-awq | agentic | 4k | 543243 | 178 | 420 | 2.36 | 21% | 5 |
| gemma-3-27b-it-int4-awq | agentic | whole | 543243 | 176 | 436 | 2.48 | 22% | 5 |
| gemma-3-27b-it-int4-awq | code | 4k | 1127490 | 140 | 393 | 2.81 | 24% | 5 |
| gemma-3-27b-it-int4-awq | code | whole | 1127490 | 141 | 406 | 2.89 | 24% | 5 |
| gemma-3-27b-it-int4-awq | english | 4k | 114363 | 589 | 510 | 0.87 | 22% | 5 |
| gemma-3-27b-it-int4-awq | english | whole | 114363 | 575 | 530 | 0.92 | 24% | 5 |
| gpt-oss-20b | agentic | 4k | 72163 | 4256 | 5689 | 1.34 | 1% | 5 |
| gpt-oss-20b | agentic | whole | 72163 | 4453 | 6003 | 1.35 | 3% | 5 |
| gpt-oss-20b | code | 4k | 0 | 22677 | 27558 | 1.22 | 4% | 5 |
| gpt-oss-20b | code | whole | 0 | 28439 | 32743 | 1.15 | 5% | 5 |
| gpt-oss-20b | english | 4k | 0 | 31026 | 55351 | 1.78 | 3% | 5 |
| gpt-oss-20b | english | whole | 0 | 47147 | 71475 | 1.52 | 6% | 5 |
| gpt2 | agentic | 4k | 0 | 27557 | 37736 | 1.37 | 3% | 5 |
| gpt2 | agentic | whole | 0 | 47468 | 48004 | 1.01 | 5% | 5 |
| gpt2 | code | 4k | 0 | 20488 | 24885 | 1.21 | 3% | 5 |
| gpt2 | code | whole | 0 | 27949 | 28595 | 1.02 | 4% | 5 |
| gpt2 | english | 4k | 0 | 27359 | 38092 | 1.39 | 3% | 5 |
| gpt2 | english | whole | 0 | 50593 | 48468 | 0.96 | 4% | 5 |
| surya-ocr-2 | agentic | 4k | 824820 | 156 | 177 | 1.13 | 2% | 5 |
| surya-ocr-2 | agentic | whole | 824820 | 156 | 176 | 1.12 | 1% | 5 |
| surya-ocr-2 | code | 4k | 1023452 | 173 | 174 | 1.01 | 4% | 5 |
| surya-ocr-2 | code | whole | 1023452 | 174 | 175 | 1.01 | 4% | 5 |
| surya-ocr-2 | english | 4k | 155943 | 125 | 138 | 1.10 | 1% | 5 |
| surya-ocr-2 | english | whole | 155943 | 125 | 140 | 1.12 | 1% | 5 |

## End to end: `encode` (no post-processor specials) on 4 KB pieces

| model | corpus | buckets | aho-corasick | ratio | spread | n |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | agentic | 216 | 209 | 0.97 | 2% | 5 |
| DeepSeek-V3 | code | 189 | 189 | 1.00 | 2% | 5 |
| DeepSeek-V3 | english | 164 | 164 | 1.00 | 2% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | agentic | 157 | 166 | 1.06 | 2% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | code | 151 | 164 | 1.08 | 2% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | english | 113 | 121 | 1.07 | 2% | 5 |
| Qwen3-8B | agentic | 202 | 198 | 0.98 | 3% | 5 |
| Qwen3-8B | code | 183 | 182 | 1.00 | 2% | 5 |
| Qwen3-8B | english | 151 | 150 | 0.99 | 5% | 5 |
| gemma-3-27b-it-int4-awq | agentic | 18 | 20 | 1.09 | 3% | 5 |
| gemma-3-27b-it-int4-awq | code | 17 | 20 | 1.13 | 2% | 5 |
| gemma-3-27b-it-int4-awq | english | 12 | 12 | 0.98 | 2% | 5 |
| gpt-oss-20b | agentic | 174 | 174 | 1.00 | 1% | 5 |
| gpt-oss-20b | code | 173 | 173 | 1.00 | 2% | 5 |
| gpt-oss-20b | english | 134 | 134 | 1.00 | 3% | 5 |
| gpt2 | agentic | 243 | 244 | 1.00 | 4% | 5 |
| gpt2 | code | 203 | 203 | 1.00 | 3% | 5 |
| gpt2 | english | 181 | 183 | 1.01 | 3% | 5 |
| surya-ocr-2 | agentic | 12 | 11 | 0.90 | 3% | 5 |
| surya-ocr-2 | code | 12 | 11 | 0.90 | 2% | 5 |
| surya-ocr-2 | english | 12 | 11 | 0.89 | 3% | 5 |

## Memory and build time

Matcher: bytes left allocated after building it from every added token (and the peak during the build), build time median of rounds. Tokenizer: bytes left allocated by `from_json`, both matchers and the model included.

| model | added tokens | matcher KiB buckets | matcher KiB a-c | peak KiB buckets | peak KiB a-c | build µs buckets | build µs a-c | tokenizer MiB buckets | tokenizer MiB a-c |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | 818 | 555 | 616 | 623 | 943 | 5385 | 5948 | 8.1 | 8.2 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 1000 | 34 | 75 | 105 | 251 | 253 | 660 | 9.2 | 9.2 |
| Qwen3-8B | 26 | 594 | 652 | 601 | 670 | 101 | 212 | 8.8 | 8.8 |
| gemma-3-27b-it-int4-awq | 6415 | 1206 | 1444 | 1570 | 2507 | 4768 | 12266 | 19.5 | 19.7 |
| gpt-oss-20b | 21 | 783 | 816 | 790 | 829 | 111 | 218 | 15.6 | 15.7 |
| gpt2 | 1 | 197 | 199 | 203 | 214 | 68 | 233 | 3.3 | 3.3 |
| surya-ocr-2 | 1948 | 323 | 471 | 427 | 1789 | 462 | 1968 | 9.3 | 9.4 |
