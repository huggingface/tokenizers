import json, os, random
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from transformers import AutoTokenizer

OUT = "/private/tmp/claude-501/-Users-arthurzucker-Work-tokenizers/2cfaf70f-8996-4e18-a61f-4533fac05d6e/scratchpad/chat_density.json"
random.seed(7)

CANDS = [
    ("deepseek-r1-qwen", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("deepseek-r1-llama", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ("deepseek-v3", "deepseek-ai/DeepSeek-V3"),
    ("deepseek-v2.5", "deepseek-ai/DeepSeek-V2.5"),
    ("gemma3-12b", "unsloth/gemma-3-12b-it"),
    ("gemma3-4b", "unsloth/gemma-3-4b-it"),
    ("gemma2-9b", "unsloth/gemma-2-9b-it"),
    ("gemma2-2b", "unsloth/gemma-2-2b-it"),
    ("qwen3", "Qwen/Qwen3-8B"),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwq", "Qwen/QwQ-32B"),
    ("llama3.3", "unsloth/Llama-3.3-70B-Instruct"),
    ("llama3.1", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
    ("llama3", "unsloth/llama-3-8b-Instruct"),
    ("mistral-nemo", "mistralai/Mistral-Nemo-Instruct-2407"),
    ("ministral8b", "mistralai/Ministral-8B-Instruct-2410"),
    ("mistral-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("phi4", "microsoft/phi-4"),
    ("phi3.5", "microsoft/Phi-3.5-mini-instruct"),
    ("smollm3", "HuggingFaceTB/SmolLM3-3B"),
    ("smollm2", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("olmo2", "allenai/OLMo-2-1124-7B-Instruct"),
    ("granite3.1", "ibm-granite/granite-3.1-8b-instruct"),
    ("qwen2.5-vl", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("smolvlm", "HuggingFaceTB/SmolVLM-Instruct"),
    ("pixtral", "mistral-community/pixtral-12b"),
]

WORDS = ("the model reasons about each token then decides the next step carefully while scanning bytes "
         "through the input stream considering many alternatives before committing to an answer that is "
         "concise accurate and grounded in the provided context without unnecessary repetition or filler").split()

def filler(n):
    return " ".join(random.choice(WORDS) for _ in range(max(1, n)))

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

LENS = [2, 8, 32, 128, 512, 2048]  # words per message -> sweeps how long plain text dwells between specials

def render(tok, L):
    msgs_full = [
        {"role": "system", "content": filler(max(2, L // 4))},
        {"role": "user", "content": filler(L)},
        {"role": "assistant", "content": filler(L)},
        {"role": "user", "content": filler(L)},
    ]
    msgs_nosys = msgs_full[1:]
    for msgs in (msgs_full, msgs_nosys, [{"role": "user", "content": filler(L)}]):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            continue
    return None

models = []
for short, mid in CANDS:
    try:
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    except Exception as e:
        print(f"skip {short}: load {repr(e)[:80]}"); continue
    if not getattr(tok, "chat_template", None):
        print(f"skip {short}: no chat_template"); continue
    sp = specials(tok)
    prompts = []
    for L in LENS:
        s = render(tok, L)
        if s:
            prompts.append(s)
    if len(prompts) < 3:
        print(f"skip {short}: template render failed"); continue
    models.append({"name": short, "specials": sp, "prompts": prompts})
    print(f"OK {short:16} specials={len(sp):<5} prompts={len(prompts)}  bytes {len(prompts[0])}..{len(prompts[-1])}")

json.dump(models, open(OUT, "w"))
print(f"\nWROTE {OUT}  models={len(models)}")
