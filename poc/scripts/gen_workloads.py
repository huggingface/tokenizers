import json, os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from transformers import AutoTokenizer

BASE = "/private/tmp/claude-501/-Users-arthurzucker-Work-tokenizers/2cfaf70f-8996-4e18-a61f-4533fac05d6e/scratchpad"
OUT = f"{BASE}/workloads.json"

CHAT = [
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
    ("mistral-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("phi4", "microsoft/phi-4"),
    ("phi3.5", "microsoft/Phi-3.5-mini-instruct"),
    ("smollm3", "HuggingFaceTB/SmolLM3-3B"),
    ("smollm2", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("olmo2", "allenai/OLMo-2-1124-7B-Instruct"),
    ("granite3.1", "ibm-granite/granite-3.1-8b-instruct"),
    ("qwen2.5-vl", "Qwen/Qwen2.5-VL-7B-Instruct"),
]
# vision: model -> processor id (image workload only). gemma3 is multimodal.
VLM = {
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "gemma3-4b": "unsloth/gemma-3-4b-it",
    "gemma3-12b": "unsloth/gemma-3-12b-it",
}

# real corpora -> realistic byte distributions (code has '<','>' decoys; doc is prose+markdown)
CODE = open(f"{BASE}/iree-core/runtime/src/iree/tokenizer/tokenizer.c").read()[:16000]
DOC = open(f"{BASE}/iree-core/runtime/src/iree/tokenizer/README.md").read()[:12000]
REASON = (
    "Let me work through this carefully. First I restate the problem and identify the constraints. "
    "The key quantity is the throughput of the scan, which depends on how often a candidate byte appears. "
    "If specials are rare, a SIMD skip dominates and the per-byte automaton is wasteful; if specials are "
    "dense, the automaton's flat cost wins. I should bound each case and compare. Consider the sparse regime: "
    "the scanner jumps between candidates, doing O(1) work per candidate and almost nothing in between. "
    "Now the dense regime: every few bytes is a candidate, so the constant per-candidate cost accumulates. "
    "The crossover is where candidate spacing roughly equals the per-candidate work divided by the per-byte "
    "automaton step. I will estimate both and sanity-check against the measured numbers before concluding. "
)
REASON_LONG = (REASON * 24)            # ~16KB chain-of-thought
DIALOG = [
    ("user", "Hey, can you explain what a tokenizer special token is?"),
    ("assistant", "Sure — it's a reserved string like a turn marker or an image placeholder that maps to a fixed id and isn't split by the model."),
    ("user", "And why do they need special matching?"),
    ("assistant", "Because they must be matched as whole units before normal subword tokenization, otherwise they'd be broken apart."),
    ("user", "Got it, thanks!"),
]
DENSE_TURNS = [("user", "ok"), ("assistant", "ok"), ("user", "next?"), ("assistant", "yes"),
               ("user", "more"), ("assistant", "sure"), ("user", "again"), ("assistant", "done"),
               ("user", "and?"), ("assistant", "right"), ("user", "ok"), ("assistant", "ok"),
               ("user", "go"), ("assistant", "yep"), ("user", "fin"), ("assistant", "ok")]

def specials(tok):
    out = []
    try:
        for t in tok.added_tokens_decoder.values():
            c = getattr(t, "content", "")
            if c: out.append(c)
    except Exception: pass
    out += list(getattr(tok, "all_special_tokens", []) or [])
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen: seen.add(c); uniq.append(c)
    return uniq

def render(tok, msgs, gen=True):
    for m in (msgs, [x for x in msgs if x["role"] != "system"]):
        try: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=gen)
        except Exception: continue
    return None

def workloads(tok):
    sysm = {"role": "system", "content": "You are a precise, concise assistant."}
    out = {}
    # thinking: long reasoning transcript (lots of context, few specials)
    out["thinking"] = render(tok, [{"role": "user", "content": "Derive the crossover point and explain."},
                                   {"role": "assistant", "content": REASON_LONG},
                                   {"role": "user", "content": "Now summarize in one line."}])
    # paste: user pastes a big code file
    out["paste"] = render(tok, [sysm, {"role": "user", "content": "Review this source and find bugs:\n\n" + CODE}])
    # conv+context: RAG-style large doc in context, multi-turn
    out["conv+ctx"] = render(tok, [sysm,
                                   {"role": "user", "content": "Context:\n" + DOC + "\n\nUsing the context, how are special tokens bucketed?"},
                                   {"role": "assistant", "content": "They are grouped by first byte into buckets, each with a shared prefix."},
                                   {"role": "user", "content": "And how is the longest match chosen?"}])
    # conv: ordinary short multi-turn chat
    out["conv"] = render(tok, [sysm] + [{"role": r, "content": c} for r, c in DIALOG])
    # dense: many tiny turns -> markers dominate
    out["dense"] = render(tok, [{"role": r, "content": c} for r, c in DENSE_TURNS])
    return {k: v for k, v in out.items() if v}

def image_item(short):
    if short not in VLM: return None
    try:
        from PIL import Image
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(VLM[short], trust_remote_code=True)
        tk = getattr(proc, "tokenizer", proc)
        img = Image.new("RGB", (672, 672), (110, 180, 90))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe this image in detail."}]}]
        enc = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True)
        ids = enc["input_ids"]
        ids = ids[0] if hasattr(ids, "__len__") and not isinstance(ids[0], int) else ids
        return tk.decode(list(ids), skip_special_tokens=False)
    except Exception as e:
        print(f"  img skip {short}: {repr(e)[:80]}")
        return None

models = []
for short, mid in CHAT:
    try:
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    except Exception as e:
        print(f"skip {short}: {repr(e)[:70]}"); continue
    if not getattr(tok, "chat_template", None):
        print(f"skip {short}: no chat_template"); continue
    sp = specials(tok)
    items = [{"type": t, "text": s} for t, s in workloads(tok).items()]
    im = image_item(short)
    if im: items.append({"type": "image", "text": im})
    if len(items) < 3:
        print(f"skip {short}: too few workloads"); continue
    models.append({"name": short, "specials": sp, "items": items})
    print(f"OK {short:16} specials={len(sp):<5} types={[i['type'] for i in items]}")

json.dump(models, open(OUT, "w"))
print(f"\nWROTE {OUT}  models={len(models)}")
