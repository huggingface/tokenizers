<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/huggingface/tokenizers/actions"><img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg"></a>
    <a href="https://crates.io/crates/tokenizers"><img alt="Crates.io" src="https://img.shields.io/crates/v/tokenizers.svg"></a>
    <a href="https://docs.rs/tokenizers/"><img alt="Docs" src="https://docs.rs/tokenizers/badge.svg"></a>
    <a href="#footprint"><img alt="Size" src="https://img.shields.io/badge/gzipped-325%20KB-brightgreen"></a>
    <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue&cachedrop"></a>
</p>

<p align="center">
    <a href="README.md">English</a> · <b>简体中文</b>
</p>

全语言、全模型、全硬件平台的最前沿（SOTA）分词库。

我们在 `tokenizers` 上的目标是开发并维护业界标准的分词引擎，使其成为所有人为整个大模型生态做出贡献的绝对核心。

### 候选发布版本：v1.0.0.rc.0

随着我们从 0.23 迁移至 v1.0.0，当前的库并未包含所有特性。如果您有所顾虑，请查看 [v1 特性与路线图](#v100-路线图) 部分，了解我们正在恢复的内容。
我们将发布一篇关于引入的重大破坏性改动（breaking changes）的博客。我们已竭尽全力将这些改动保持在最小限度。

# 安装指南

```bash
pip install --pre tokenizers
```

# 使用方法

```python
>>> from tokenizers import Tokenizer
>>> tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B")   # 或通过 .from_file(path) 加载
>>> tokenizer.encode("Hello, y'all! How are you 😁 ?")
Encoding(ids=[128000, 9906, 11, 379, 65948, 0, 2650, 527, 499, 27623, 223, 949], type_ids=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], attention_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
>>> tokenizer.tokenize("Hello")
['<|begin_of_text|>', 'Hello']
>>> tokenizer.decode([15339, 1917])
'hello world'
>>> tokenizer.decode_tokens(tokenizer.encode("Hello"))
['<|begin_of_text|>', 'Hello']
```

# 性能表现

<img width="1238" height="677" alt="性能基准评测" src="https://github.com/user-attachments/assets/49c672aa-f7c3-4eb5-a89c-99703f8ee38f" />

更多基准评测细节，请查看 https://huggingface-tokenizers-v1.static.hf.space/index.html。

# 多语言绑定支持

| 语言 | 状态 |
|---|---|
| [Rust](tokenizers) | ✅ 官方参考实现 |
| [Python](bindings/python) | ✅ |
| [Node.js](bindings/node) | ✅ |
| C / C++ / Java / Go 等 | 🚧 计划中 |
| [Ruby](https://github.com/ankane/tokenizers-ruby) | 社区外部仓库维护 |

# v1.0.0 路线图

在我们迈向 v1 版本的过程中，我们将全面恢复支持 `transformers` 风格与分词器对象交互的完整 Python API：

## 待办清单 (TODOS):

### Python 绑定 (Python Bindings)

- 恢复训练功能：实现难度不高，但我们希望性能提升同步到位。
- 批量编码 API，支持编码 ID 的零拷贝（zero-copy）视图。
- 用于 PyTorch / JAX / NumPy 互操作的 dlpack API。
- 组合式设计：灵活编辑流水线各组件（例如 `tokenizer.normalizer = [...]`）。
- `tokenizer.save`：支持将修改或训练后的分词器持久化存储至磁盘。
- `role_to_token` 角色映射覆盖。
- `encode` / `encode_batch` 支持输入文本对（pair input）。
- 恢复 offset 偏移量输出：这对训练阶段非常实用。
- 编码中的滑动窗口与步幅支持（Windowing / stride）。
- 流式解码支持（`decode_stream`）。

### Rust 核心 (Rust core)

- `bitnorm`：通过优化规范化实现更高性能，由于这会引入破坏性改动，我们希望确保其纳入 v1。
- 滑动窗口（带步幅的截断）。
- 优化训练性能。
- 优化解码性能。
- 字符偏移量追踪（Offset tracking）。
- 填充 Padding：2D 扁平缓冲区。
- 词缓存（Wordcache）：`shrn` 与模拟 movemask 对比。
- 修复 ModernBERT / OLMo 加载失败问题（“64 位哈希冲突”）。
- 展开更多正则表达式：覆盖最高频的规则，避免遗漏。

### 其他 (Other)

- 针对加载、编码与解码 API 的 C & C++ 绑定。

## 1.0.0 之后规划

- 优化/重写 `bitcannon` crate。
- 基于 C 绑定扩展 Java、Go、Swift... 等多语言绑定。
- `tk-devices` —— 探索性 GPU 编码与批量解码技术，将文本和 Token ID 保留在设备显存中：单次上传词表，并行计算输出位置，并在 GPU 上聚合字节。这是一个面向大批次的探索性可选组件，取决于原型设计与基准测试结果。

<a name="footprint"></a>

# Crate 架构详情

## Crate 体积

<img width="784" height="945" alt="Crate 体积分析" src="https://github.com/user-attachments/assets/3cd1d537-2a39-4f1c-8863-c79c28c880a7" />

## 子 Crates 模块

<details>
<summary><b><code>tk-encode</code></b> — 推理引擎</summary>

包含各种分词模型（BPE、Unigram、WordPiece、WordLevel）与完整的流水线设计：`Normalizer`（规范器）、`PreTokenizer`（前置分词器）、`Model`（模型核心）、`PostProcessor`（后处理器）、`Decoder`（解码器）。

**为什么独立：** 在生产环境中部署模型服务并不需要全部功能。将其与训练拆分后，服务端二进制文件无需链接训练器、语料读取器或旧版转换工具等无用依赖。
</details>

<details>
<summary><b><code>bitcannon</code></b> — 比特流前置分词</summary>

Unicode 标签分类与基于**比特流程序**的前置分词。参考论文 *Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs* (MICRO'25, [10.1145/3725843.3756052](https://doi.org/10.1145/3725843.3756052))，但我们加入了自研算法以确保无需依赖复杂的 parabix 引擎。
字符首先被分类为“标签”，使用编译 `bitcannon` 时生成的稀疏查找表。这意味着对沉重的 Unicode 数据表做到了 0 依赖。

内置了针对 gpt2 / ByteLevel、cl100k、o200k、tekken、deepseek 和 kimi-k2 的精确字节文法。
各类别或标签分别具有高半字节和低半字节（编码在 u8 上），并在需要时携带精细化信息。使用 u8 标签大幅减少比特流处理开销，让前置分词更加简捷极速。

**为什么独立：** 我们认为它未来可能对更多项目有所裨益。据我们所知，在 CPU 上针对所有前置分词正则表达式，该实现方案均达到了极致的速度。
</details>

<details>
<summary><b><code>tk-serialize</code></b> — 解析读取器</summary>

`from_json_file` 将规范的 `tokenizer.json` 转换为 `PipelineTokenizer`。

**为什么独立：** 在某些定制场景下可以更轻量地按需引入。
</details>

<details>
<summary><b><code>tk-convert</code></b> — 升级转换通道</summary>

`canonicalize_file` 将本库任意旧版本生成的旧版 `tokenizer.json` 重写为当前读取器接受的规范形式。纯粹的 JSON→JSON 转换，仅依赖 `std::path` 与 `serde_json`。

**为什么独立：** 历史上发布的所有配置文件都能被顺利读取，同时运行时无需携带长达十年的兼容性分支代码。旧版兼容代码占据了最终 crate 体积相当大的比例，禁用它可以让端侧部署构建更为轻巧。
</details>

<details>
<summary><b><code>tk-train</code></b> — 训练组件</summary>

`Trainer` trait、各类具体 `*Trainer` 实现、`TrainerWrapper` 以及 `Trainable` 扩展。

**为什么独立：** 训练是工作站上的批处理任务；推理则是服务端的高频热循环。两者的约束与优化目标截然相反，因此各自拥有独立的依赖预算。
</details>

<details>
<summary><b><code>bitmap_gen</code></b> — 仅开发期使用的表生成器</summary>

`cargo run -p bitmap_gen` 基于 `unicode-properties` 重新生成 `bitcannon` 提交的分类表，为每个字符码点生成一个 `Atom` 标签。

**为什么独立 — 这是实现 Unicode 零依赖的核心关键：** `unicode-properties` 仅是该 crate 的依赖，不参与其他组件编译。它生成的表直接作为常规源码检入到 `bitcannon/src/classify/atom_tables.rs` 中。因此，Unicode 数据在开发期间一次性解析完成，该生成器 crate 绝不会链接到最终发布的产物中，也不会公开发布。同时无需 build script 脚本 —— `bitcannon` 在编译时不需要代码生成步骤，发布的二进制文件包含标签而无需携带 Unicode crate 进行推导。发布工作流会重新运行该生成器，若提交的代码表不一致则构建失败，确保预编译表绝不会悄悄过期。
</details>

<details>
<summary><b><code>tokenizers</code></b> — 聚合入口</summary>

轻量级的统一重新导出层，保持现有的 `tokenizers::…` 引用路径平滑可用。除非你明确知道需要裁剪依赖，否则通常直接依赖此 crate 即可。
</details>

## 硬件平台与 SIMD 适配

**无需强制 SIMD 支持。** 每个算子内核都包含一个始终编译的可移植路径，并作为其向量化同构实现的精确字节测试基准。因此，代码正确性绝不依赖于特定硬件内核的存在 —— 硬件内核仅影响吞吐吞吐量。

<details>
<summary>哪些算子进行了硬件加速自适应</summary>

| 算子 (op) | aarch64 | x86_64 | wasm32 | 可移植标量回退 |
|---|---|---|---|---|
| Unicode atom 分类 | NEON (基线) | AVX-512 VBMI → SSE4.1/SSSE3 (运行时探测) | SIMD128 | `classify_scalar` |
| 比特流块构建 | NEON | SSE/AVX | — | `build_block_scalar` |
| 字面量与新增 token 扫描 | NEON | x86_64 | — | scalar |
| 词表桶半字节匹配 | NEON | — | — | scalar |

aarch64 在编译期完成选型（NEON 为基线）。x86_64 在运行期通过 `is_x86_feature_detected!` 动态分发，因此单个二进制文件即可覆盖自 2008 年以来的所有 x86_64 CPU。wasm32 需要开启 `simd128` 目标特性；若未开启则自动回退至标量遍历。
</details>

# 致谢与开源传承

本项目的打造站在了众多优秀开源项目的肩膀之上。由衷感谢所有项目的杰出贡献：

[gigatoken](https://github.com/marcelroed/gigatoken)、[tiktoken](https://crates.io/crates/tiktoken-rs)、[kitoken](https://crates.io/crates/kitoken)、[tokie](https://crates.io/crates/tokie)、[fastokens](https://crates.io/crates/fastokens)、[wordchipper](https://crates.io/crates/wordchipper) 以及 [ai-tokenizer](https://www.npmjs.com/package/ai-tokenizer) 各自都在快速分词器的边界上做出了积极探索；我们深入研读了这些工作，上述的诸多核心设计理念也正是因为这些项目的先行验证而得以融入本项目。

官方文档：[快速指南](https://huggingface.co/docs/tokenizers/index) ·
[快速上手](https://huggingface.co/docs/tokenizers/quicktour) ·
[docs.rs](https://docs.rs/tokenizers/)

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年10月10日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
