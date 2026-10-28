# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers==4.57.1", "jinja2>=3.1", "pyarrow>=17", "huggingface_hub>=0.34"]
# ///
"""Write the benchmark corpora and tokenizers to <out>:

    english.txt          FineWeb sample-10BT, documents appended until CORPUS_BYTES
    code.txt             SWE-smith trajectories' `patch` column (unified diffs of Python repos), deduplicated
    agentic_<model>.txt  SWE-smith `tool` trajectories through <model>'s own chat template
    <model>.json         <model>'s tokenizer.json
    sets.tsv             model, tokenizer.json, corpus name, corpus path: one row per benchmark cell
    revisions.json       every repo revision used

A model without a chat template gets the trajectories as `role: content` lines instead.
"""
import json
import os
import pathlib
import sys
import time

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

CORPUS_BYTES = int(os.environ.get("CORPUS_BYTES", str(20 << 20)))
FINEWEB = ("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet")
TRACES = ("SWE-bench/SWE-smith-trajectories", [f"data/tool-0000{i}-of-00008.parquet" for i in range(8)])
# Few added tokens first, then many. Every repo is public and ungated.
MODELS = [
    "openai-community/gpt2",                    # 1
    "openai/gpt-oss-20b",                       # ~21
    "Qwen/Qwen3-8B",                            # ~26
    "deepseek-ai/DeepSeek-V3",                  # ~818
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",  # 1000
    "datalab-to/surya-ocr-2",                   # ~1948, 40 distinct first bytes
    "gaunernst/gemma-3-27b-it-int4-awq",        # ~6415, Gemma 3's tokenizer
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(repo, repo_type="model"):
    info = HfApi().repo_info(repo, repo_type=repo_type)
    return info.sha


def parquet_rows(repo, rev, paths, column):
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    for path in paths:
        with fs.open(f"datasets/{repo}@{rev}/{path}", block_size=8 << 20) as f:
            pf = pq.ParquetFile(f)
            for group in range(pf.num_row_groups):
                yield from pf.read_row_group(group, columns=[column]).column(column).to_pylist()


def fill(rows, limit):
    parts, size = [], 0
    for text in rows:
        parts.append(text)
        size += len(text.encode())
        if size >= limit:
            break
    return "".join(parts), size


def messages(row):
    """SWE-smith's OpenAI-style messages in the shape every chat template accepts: plain-text content and tool-call
    arguments as dicts."""
    out = []
    for m in json.loads(row) if isinstance(row, str) else row:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        msg = {"role": m["role"], "content": content}
        if m.get("tool_calls"):
            msg["tool_calls"] = [{"type": "function", "id": c.get("id"),
                                  "function": {"name": c["function"]["name"],
                                               "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                                 for c in m["tool_calls"]]
        if m["role"] == "tool" and m.get("tool_call_ids"):
            msg["tool_call_id"] = m["tool_call_ids"][0]
        out.append(msg)
    return out


def alternating(conv):
    """The conversation as alternating user/assistant turns, for templates that accept nothing else: system and tool
    messages become user text, tool calls become assistant text, and consecutive turns of one role are merged."""
    turns = []
    for m in conv:
        role = "assistant" if m["role"] == "assistant" else "user"
        text = m["content"] + "".join(f"\n{json.dumps(c['function'])}" for c in m.get("tool_calls", []))
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + text
        else:
            turns.append({"role": role, "content": text})
    if turns and turns[0]["role"] == "assistant":
        turns.insert(0, {"role": "user", "content": ""})
    return turns


def chat_tokenizer(model, rev, tokenizer_json):
    """The model's tokenizer for its chat template. A tokenizer class this transformers lacks falls back to the bare
    tokenizer.json with the repo's template."""
    from huggingface_hub.errors import EntryNotFoundError
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(model, revision=rev)
    except ValueError as e:
        log(f"{model}: {e}; using tokenizer.json and the repo's template")
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
    config = json.loads(pathlib.Path(hf_hub_download(model, "tokenizer_config.json", revision=rev)).read_text())
    try:
        tok.chat_template = pathlib.Path(hf_hub_download(model, "chat_template.jinja", revision=rev)).read_text()
    except EntryNotFoundError:
        tok.chat_template = config.get("chat_template")
    return tok


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    revisions = {}

    repo, path = FINEWEB
    revisions[repo] = rev = sha(repo, "dataset")
    text, size = fill((t + "\n" for t in parquet_rows(repo, rev, [path], "text")), CORPUS_BYTES)
    (out / "english.txt").write_text(text)
    log(f"english: {size} bytes from {repo}@{rev[:8]}")

    repo, paths = TRACES
    revisions[repo] = rev = sha(repo, "dataset")
    patches = dict.fromkeys(p for p in parquet_rows(repo, rev, paths, "patch") if p)
    text, size = fill(patches, CORPUS_BYTES)
    (out / "code.txt").write_text(text)
    log(f"code: {size} bytes, {len(patches)} distinct patches from {repo}@{rev[:8]}")

    convs = []
    for conv in parquet_rows(repo, rev, paths, "messages"):
        convs.append(messages(conv))
        if len(convs) >= 6000:
            break
    log(f"traces: {len(convs)} conversations")

    rows = []
    for model in MODELS:
        name = model.split("/")[1]
        revisions[model] = rev = sha(model)
        hf_hub_download(model, "tokenizer.json", revision=rev, local_dir=out / name)
        (out / name / "tokenizer.json").rename(out / f"{name}.json")
        tok = chat_tokenizer(model, rev, out / f"{name}.json")
        rendered, retried, failed = [], 0, 0
        for conv in convs:
            if tok.chat_template:
                try:
                    rendered.append(tok.apply_chat_template(conv, tokenize=False))
                    continue
                except Exception:  # noqa: BLE001  a template that rejects this conversation's shape
                    pass
                try:
                    rendered.append(tok.apply_chat_template(alternating(conv), tokenize=False))
                    retried += 1
                    continue
                except Exception:  # noqa: BLE001
                    failed += 1
            rendered.append("".join(f"{m['role']}: {m['content']}\n" for m in conv))
        text, size = fill(rendered, CORPUS_BYTES)
        (out / f"agentic_{name}.txt").write_text(text)
        added = len(json.loads((out / f"{name}.json").read_text()).get("added_tokens", []))
        log(f"{model}@{rev[:8]}: {added} added tokens, agentic {size} bytes, template "
            f"{'yes' if tok.chat_template else 'no'}, {retried} conversations rendered as alternating turns, "
            f"{failed} rejected (written as plain lines)")
        for corpus in ["english", "code", f"agentic_{name}"]:
            rows.append(f"{name}\t{out}/{name}.json\t{corpus.split('_')[0]}\t{out}/{corpus}.txt\n")

    (out / "sets.tsv").write_text("".join(rows))
    (out / "revisions.json").write_text(json.dumps(revisions, indent=1))
    print("== revisions " + json.dumps(revisions), flush=True)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
