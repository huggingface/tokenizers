# Added-token matcher, job 6ae20e183fb22a235dd41773

- machine AMD EPYC 7R13 Processor, nproc 32, kernel 6.12.113-138.209.amzn2023.x86_64, pinned to taskset -c 0
- branch: main + branch.diff (142 added lines): perf/aho-corasick @ d9a5e8af+dirty
- branch Cargo.lock: version = "1.1.5" version = "2.8.3" 
- 15:18:04 end, steal ticks 39

Throughput is input MB/s, one thread, the best of the rounds (n = rounds per arm). Ratio = aho-corasick / buckets: above 1 means aho-corasick is faster.
Spread = (max - min) / max over the rounds, worst arm.

Identical outputs between arms: all cells

## Matcher alone: every match, restarting after each, like `extract_next`

| model | corpus | calls | matches | buckets | aho-corasick | ratio | spread | n |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | agentic | 4k | 32929 | 5896 | 5241 | 0.89 | 5% | 5 |
| DeepSeek-V3 | agentic | whole | 32929 | 6170 | 5505 | 0.89 | 2% | 5 |
| DeepSeek-V3 | code | 4k | 0 | 24875 | 26222 | 1.05 | 8% | 5 |
| DeepSeek-V3 | code | whole | 0 | 31763 | 30698 | 0.97 | 6% | 5 |
| DeepSeek-V3 | english | 4k | 0 | 47116 | 53678 | 1.14 | 4% | 5 |
| DeepSeek-V3 | english | whole | 0 | 70828 | 72513 | 1.02 | 7% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | agentic | 4k | 65998 | 1407 | 3411 | 2.43 | 5% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | agentic | whole | 65998 | 1415 | 3563 | 2.52 | 5% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | code | 4k | 1 | 1651 | 8549 | 5.18 | 2% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | code | whole | 1 | 1648 | 8814 | 5.35 | 1% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | english | 4k | 0 | 1738 | 42396 | 24.39 | 3% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | english | whole | 0 | 1772 | 52513 | 29.63 | 4% | 5 |
| Qwen3-8B | agentic | 4k | 53562 | 4908 | 6439 | 1.31 | 1% | 5 |
| Qwen3-8B | agentic | whole | 53562 | 5127 | 6701 | 1.31 | 1% | 5 |
| Qwen3-8B | code | 4k | 0 | 24402 | 26355 | 1.08 | 7% | 5 |
| Qwen3-8B | code | whole | 0 | 30801 | 31233 | 1.01 | 4% | 5 |
| Qwen3-8B | english | 4k | 0 | 46874 | 52821 | 1.13 | 3% | 5 |
| Qwen3-8B | english | whole | 0 | 69770 | 69358 | 0.99 | 5% | 5 |
| gemma-3-27b-it-int4-awq | agentic | 4k | 543243 | 177 | 418 | 2.36 | 5% | 5 |
| gemma-3-27b-it-int4-awq | agentic | whole | 543243 | 176 | 433 | 2.46 | 2% | 5 |
| gemma-3-27b-it-int4-awq | code | 4k | 1127490 | 140 | 391 | 2.80 | 5% | 5 |
| gemma-3-27b-it-int4-awq | code | whole | 1127490 | 140 | 408 | 2.91 | 4% | 5 |
| gemma-3-27b-it-int4-awq | english | 4k | 114363 | 588 | 527 | 0.90 | 5% | 5 |
| gemma-3-27b-it-int4-awq | english | whole | 114363 | 569 | 529 | 0.93 | 1% | 5 |
| gpt-oss-20b | agentic | 4k | 72163 | 4226 | 5634 | 1.33 | 2% | 5 |
| gpt-oss-20b | agentic | whole | 72163 | 4462 | 5976 | 1.34 | 2% | 5 |
| gpt-oss-20b | code | 4k | 0 | 22620 | 26760 | 1.18 | 14% | 5 |
| gpt-oss-20b | code | whole | 0 | 28121 | 32277 | 1.15 | 8% | 5 |
| gpt-oss-20b | english | 4k | 0 | 30962 | 53705 | 1.73 | 5% | 5 |
| gpt-oss-20b | english | whole | 0 | 47765 | 70010 | 1.47 | 6% | 5 |
| gpt2 | agentic | 4k | 0 | 27657 | 36744 | 1.33 | 4% | 5 |
| gpt2 | agentic | whole | 0 | 47453 | 45790 | 0.96 | 3% | 5 |
| gpt2 | code | 4k | 0 | 20509 | 25314 | 1.23 | 13% | 5 |
| gpt2 | code | whole | 0 | 28803 | 27964 | 0.97 | 8% | 5 |
| gpt2 | english | 4k | 0 | 27599 | 37053 | 1.34 | 2% | 5 |
| gpt2 | english | whole | 0 | 50082 | 47096 | 0.94 | 3% | 5 |
| surya-ocr-2 | agentic | 4k | 824820 | 156 | 180 | 1.15 | 5% | 5 |
| surya-ocr-2 | agentic | whole | 824820 | 157 | 184 | 1.17 | 7% | 5 |
| surya-ocr-2 | code | 4k | 1023452 | 173 | 169 | 0.97 | 2% | 5 |
| surya-ocr-2 | code | whole | 1023452 | 173 | 176 | 1.01 | 5% | 5 |
| surya-ocr-2 | english | 4k | 155943 | 125 | 140 | 1.11 | 3% | 5 |
| surya-ocr-2 | english | whole | 155943 | 125 | 140 | 1.12 | 1% | 5 |

## End to end: `encode` (no post-processor specials) on 4 KB pieces

| model | corpus | buckets | aho-corasick | ratio | spread | n |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | agentic | 215 | 208 | 0.97 | 3% | 5 |
| DeepSeek-V3 | code | 188 | 188 | 1.00 | 2% | 5 |
| DeepSeek-V3 | english | 164 | 164 | 1.00 | 4% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | agentic | 156 | 166 | 1.07 | 3% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | code | 150 | 162 | 1.08 | 2% | 5 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | english | 113 | 121 | 1.07 | 4% | 5 |
| Qwen3-8B | agentic | 200 | 199 | 1.00 | 2% | 5 |
| Qwen3-8B | code | 183 | 184 | 1.01 | 3% | 5 |
| Qwen3-8B | english | 150 | 152 | 1.01 | 3% | 5 |
| gemma-3-27b-it-int4-awq | agentic | 18 | 20 | 1.09 | 1% | 5 |
| gemma-3-27b-it-int4-awq | code | 17 | 19 | 1.11 | 2% | 5 |
| gemma-3-27b-it-int4-awq | english | 12 | 12 | 0.98 | 2% | 5 |
| gpt-oss-20b | agentic | 173 | 172 | 0.99 | 2% | 5 |
| gpt-oss-20b | code | 173 | 172 | 0.99 | 3% | 5 |
| gpt-oss-20b | english | 134 | 133 | 1.00 | 3% | 5 |
| gpt2 | agentic | 243 | 242 | 1.00 | 2% | 5 |
| gpt2 | code | 202 | 203 | 1.00 | 2% | 5 |
| gpt2 | english | 180 | 182 | 1.01 | 2% | 5 |
| surya-ocr-2 | agentic | 12 | 13 | 1.03 | 2% | 5 |
| surya-ocr-2 | code | 12 | 13 | 1.02 | 2% | 5 |
| surya-ocr-2 | english | 12 | 12 | 1.02 | 2% | 5 |

## Memory and build time

Matcher: bytes left allocated after building it from every added token (and the peak during the build), build time median of rounds. Tokenizer: bytes left allocated by `from_json`, both matchers and the model included.

| model | added tokens | matcher KiB buckets | matcher KiB a-c | peak KiB buckets | peak KiB a-c | build µs buckets | build µs a-c | tokenizer MiB buckets | tokenizer MiB a-c |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | 818 | 555 | 616 | 623 | 943 | 5465 | 5974 | 8.1 | 8.2 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 1000 | 34 | 75 | 105 | 251 | 254 | 662 | 9.2 | 9.2 |
| Qwen3-8B | 26 | 594 | 652 | 601 | 670 | 105 | 214 | 8.8 | 8.8 |
| gemma-3-27b-it-int4-awq | 6415 | 1206 | 1444 | 1570 | 2507 | 4723 | 12303 | 19.5 | 19.7 |
| gpt-oss-20b | 21 | 783 | 816 | 790 | 829 | 114 | 222 | 15.6 | 15.7 |
| gpt2 | 1 | 197 | 199 | 203 | 214 | 67 | 231 | 3.3 | 3.3 |
| surya-ocr-2 | 1948 | 323 | 471 | 427 | 1789 | 469 | 1939 | 9.3 | 9.4 |
