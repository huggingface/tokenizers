#!/usr/bin/env python3
"""Fetch the 22 tokenizer.json files used by the `fast-encode` SWEEP into poc/data/toks/.
Requires: pip install huggingface_hub  (and `huggingface-cli login` for any gated repos).
Usage: python poc/scripts/download_tokenizers.py
"""
import os, json
from huggingface_hub import hf_hub_download

MODELS = [
    ("deepseek-r1-qwen", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("deepseek-r1-llama", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ("deepseek-v3", "deepseek-ai/DeepSeek-V3"),
    ("deepseek-v2.5", "deepseek-ai/DeepSeek-V2.5"),
    ("gemma3-12b", "unsloth/gemma-3-12b-it"), ("gemma3-4b", "unsloth/gemma-3-4b-it"),
    ("gemma2-9b", "unsloth/gemma-2-9b-it"), ("gemma2-2b", "unsloth/gemma-2-2b-it"),
    ("qwen3", "Qwen/Qwen3-8B"), ("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct"), ("qwq", "Qwen/QwQ-32B"),
    ("llama3.3", "unsloth/Llama-3.3-70B-Instruct"),
    ("llama3.1", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
    ("llama3", "unsloth/llama-3-8b-Instruct"),
    ("mistral-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("phi4", "microsoft/phi-4"), ("phi3.5", "microsoft/Phi-3.5-mini-instruct"),
    ("smollm3", "HuggingFaceTB/SmolLM3-3B"), ("smollm2", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("olmo2", "allenai/OLMo-2-1124-7B-Instruct"), ("granite3.1", "ibm-granite/granite-3.1-8b-instruct"),
    ("qwen2.5-vl", "Qwen/Qwen2.5-VL-7B-Instruct"),
]
out = os.path.join(os.path.dirname(__file__), "..", "data", "toks")
os.makedirs(out, exist_ok=True)
for short, repo in MODELS:
    dst = os.path.join(out, f"{short}.json")
    if os.path.exists(dst):
        print("skip", short); continue
    try:
        p = hf_hub_download(repo, "tokenizer.json")
        open(dst, "wb").write(open(p, "rb").read())
        print("ok  ", short)
    except Exception as e:
        print("ERR ", short, repo, str(e)[:80])
