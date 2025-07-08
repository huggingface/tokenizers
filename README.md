<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
</p>

<p align="center">
    <img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg">
    <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue&cachedrop">
    </a>
    <a href="https://pepy.tech/project/tokenizers">
        <img src="https://pepy.tech/badge/tokenizers/week" />
    </a>
</p>

# ⚡ faster-whitespace-pretok

**This is a performance fork of Hugging Face’s `tokenizers`**, focused on optimizing the `Whitespace` PreTokenizer.  
It preserves all original functionality and directory layout of `tokenizers/tokenizers` for compatibility — including benchmark support and test coverage.

> 🔧 Pull Request: [huggingface/tokenizers#1822](https://github.com/huggingface/tokenizers/pull/1822)

---

## 🚀 What’s New in This Fork?

### ✅ Optimized `Whitespace` PreTokenizer
- Replaced regex-based logic with a cache-efficient manual traversal using `char_indices()`.
- No change to output behavior — identical span offsets and splits.
- Drop-in compatible with all existing pipelines.

### ✅ Criterion Benchmark Added
- Added `benches/whitespace_bench.rs`
- Measures short, medium, and long inputs
- Registered in `Cargo.toml`:

```toml
[[bench]]
name = "whitespace_bench"
harness = false
```

### ✅ Additional Variant: `WhitespaceSplit`

* Lightweight alternative that only splits on whitespace (no span tracking).
* Useful for standalone benchmarking or ultra-fast preprocessing.

---

## 📊 Benchmarks

Benchmarked using Criterion across 5 test cycles:

| Input Type | Avg. Time (Original) | Avg. Time (Optimized) | Speedup  |
| ---------- | -------------------- | --------------------- | -------- |
| Short      | \~620 ns             | \~555 ns              | ✅ 10–15% |
| Medium     | 4.3 µs               | 3.7–4.0 µs            | ✅ 5–30%  |
| Long       | \~60–74 µs           | \~50–63 µs            | ✅ 5–15%  |

* 🔬 Output remains identical to the original `Whitespace` implementation.
* 🧪 Verified with robust unit tests.
* 🔁 Consistent results across runs.

---

## 🧠 Technical Highlights

* ❌ No regex (avoids unnecessary overhead)
* ✅ Manual `char_indices()` loop for precision and cache-friendliness
* 🧠 Inline span classification
* 💡 Zero additional dependencies
* 🔄 Fully backwards-compatible with `impl_serde_type!`

---

## 📎 Related Issue

Improves local benchmarking infrastructure and test coverage related to:
[#1820](https://github.com/huggingface/tokenizers/issues/1820)

This PR does not fix dataset download issues directly, but **adds independent, reproducible local benchmarking support**.

---

## 🔧 Installation & Usage

Clone the fork and use it as a **drop-in `tokenizers/tokenizers` replacement**:

```bash
git clone https://github.com/8ria/faster-whitespace-pretok
cd faster-whitespace-pretok/tokenizers
cargo bench --bench whitespace_bench
```

Use your own sample inputs by editing `whitespace_bench.rs`.

---

## 📦 Python Installation (from this fork)

To use the Python bindings with the optimized version:

```bash
pip install git+https://github.com/8ria/faster-whitespace-pretok.git#subdirectory=bindings/python
```

> All Python-facing behavior remains identical to upstream `tokenizers`.

---

## 🙌 Why This Matters

Whitespace pre-tokenization is executed millions of times in ML workflows:

* LLM inference
* Prompt batching
* Offline training pipelines

Even small improvements in this phase **compound at scale** — especially when parallelized.

This fork improves efficiency **without changing outputs or APIs**.

---

## 📫 Contact

**AndriaK**
📧 [hey@andriaK.com](mailto:hey@andriaK.com)
🔗 [GitHub](https://github.com/8ria)
