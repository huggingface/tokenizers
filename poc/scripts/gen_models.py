import json, os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from transformers import AutoTokenizer

OUT = "/private/tmp/claude-501/-Users-arthurzucker-Work-tokenizers/2cfaf70f-8996-4e18-a61f-4533fac05d6e/scratchpad/models.json"

# (short name, model id) — modern first (DeepSeek, Gemma-2/3, Qwen3, Llama-3.3, Mistral-Nemo,
# Phi-4, SmolLM3, OLMo2, Granite, Command-R7B, modern VLMs), then a few classics for range.
CANDS = [
    ("deepseek-r1-qwen", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("deepseek-r1-llama", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ("deepseek-v3", "deepseek-ai/DeepSeek-V3"),
    ("deepseek-v2.5", "deepseek-ai/DeepSeek-V2.5"),
    ("deepseek-coder", "deepseek-ai/deepseek-coder-6.7b-instruct"),
    ("gemma3-12b", "unsloth/gemma-3-12b-it"),
    ("gemma3-4b", "unsloth/gemma-3-4b-it"),
    ("gemma2-9b", "unsloth/gemma-2-9b-it"),
    ("gemma2-2b", "unsloth/gemma-2-2b-it"),
    ("qwen3", "Qwen/Qwen3-8B"),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwq", "Qwen/QwQ-32B"),
    ("llama3.3", "unsloth/Llama-3.3-70B-Instruct"),
    ("llama3.1", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
    ("mistral-nemo", "mistralai/Mistral-Nemo-Instruct-2407"),
    ("ministral8b", "mistralai/Ministral-8B-Instruct-2410"),
    ("mistral-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("phi4", "microsoft/phi-4"),
    ("phi3.5", "microsoft/Phi-3.5-mini-instruct"),
    ("smollm3", "HuggingFaceTB/SmolLM3-3B"),
    ("smollm2", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("olmo2", "allenai/OLMo-2-1124-7B-Instruct"),
    ("granite3.1", "ibm-granite/granite-3.1-8b-instruct"),
    ("commandr7b", "CohereForAI/c4ai-command-r7b-12-2024"),
    ("qwen2.5-vl", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("smolvlm", "HuggingFaceTB/SmolVLM-Instruct"),
    ("pixtral", "mistral-community/pixtral-12b"),
    # classics for range:
    ("gpt2", "gpt2"),
    ("bert", "bert-base-uncased"),
    ("t5", "t5-base"),
    ("llama3", "unsloth/llama-3-8b-Instruct"),
]

def specials(tok):
    out = []
    try:
        for t in tok.added_tokens_decoder.values():
            c = getattr(t, "content", "")
            if c:
                out.append(c)
    except Exception:
        pass
    out += list(getattr(tok, "all_special_tokens", []) or [])
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c); uniq.append(c)
    return uniq

models, want = [], 24
for short, mid in CANDS:
    if len(models) >= want:
        break
    try:
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        sp = specials(tok)
        if not sp:
            print(f"skip {short}: 0 specials"); continue
        models.append({"name": short, "model": mid, "specials": sp})
        print(f"OK {short:12} {mid:48} specials={len(sp)}")
    except Exception as e:
        print(f"skip {short}: {repr(e)[:120]}")

json.dump(models, open(OUT, "w"))
print(f"\nWROTE {OUT}  models={len(models)}")
