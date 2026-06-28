import json, sys, traceback
from PIL import Image

OUT = "/private/tmp/claude-501/-Users-arthurzucker-Work-tokenizers/2cfaf70f-8996-4e18-a61f-4533fac05d6e/scratchpad/prompts.json"
img = Image.new("RGB", (560, 560), (123, 200, 80))

TEXT_MSGS = [
    {"role": "system", "content": "You are a helpful assistant. Be concise and accurate."},
    {"role": "user", "content": "Explain how Aho-Corasick matching works and when memchr is faster. Give a short example."},
    {"role": "assistant", "content": "Aho-Corasick builds an automaton over all patterns and scans the text once, stepping one transition per byte. memchr is faster when matches are sparse because it SIMD-skips non-candidate bytes."},
    {"role": "user", "content": "Now compare memory footprints and summarize in two sentences."},
]
VIS_MSGS = [
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image in detail, including colors and any objects."},
    ]},
]

def specials_of(tok):
    out = []
    try:
        for _id, t in sorted(tok.added_tokens_decoder.items()):
            c = getattr(t, "content", str(t))
            if c:
                out.append(c)
    except Exception:
        out = list(getattr(tok, "all_special_tokens", []))
    # de-dup preserve order
    seen = set(); uniq = []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq

def add_text(cases, name, model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    s = tok.apply_chat_template(TEXT_MSGS, tokenize=False, add_generation_prompt=True)
    cases.append({"name": name, "kind": "text", "model": model,
                  "specials": specials_of(tok), "prompt": s})
    print(f"OK  text  {name:24} len={len(s):6} specials={len(cases[-1]['specials'])}")

def add_vision(cases, name, model):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model)
    tok = getattr(proc, "tokenizer", proc)
    # tokenize=True so the processor expands image placeholders into real N-token runs,
    # then decode back to the exact string that gets re-encoded in production.
    enc = proc.apply_chat_template(VIS_MSGS, tokenize=True, add_generation_prompt=True,
                                   return_dict=True)
    ids = enc["input_ids"]
    ids = ids[0] if hasattr(ids, "__len__") and not isinstance(ids[0], int) else ids
    ids = list(ids)
    s = tok.decode(ids, skip_special_tokens=False)
    cases.append({"name": name, "kind": "image+text", "model": model,
                  "specials": specials_of(tok), "prompt": s})
    print(f"OK  vis   {name:24} len={len(s):6} specials={len(cases[-1]['specials'])} ids={len(ids)}")

cases = []
TEXT = [
    ("qwen2.5",  "Qwen/Qwen2.5-0.5B-Instruct"),
    ("llama3.1", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
    ("mistral",  "mistralai/Mistral-7B-Instruct-v0.3"),
]
VIS = [
    ("qwen2.5-vl", "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("llava1.5",   "llava-hf/llava-1.5-7b-hf"),
    ("smolvlm",    "HuggingFaceTB/SmolVLM-Instruct"),
    ("pixtral",    "mistral-community/pixtral-12b"),
]
for name, m in TEXT:
    try: add_text(cases, name, m)
    except Exception as e: print(f"SKIP text {name}: {e}")
for name, m in VIS:
    try: add_vision(cases, name, m)
    except Exception as e: print(f"SKIP vis  {name}: {repr(e)[:160]}")

# merged multi-tenant: union specials of everything loaded -> forces more buckets
allspec, seen = [], set()
for c in cases:
    for s in c["specials"]:
        if s not in seen:
            seen.add(s); allspec.append(s)
if cases:
    longest = max(cases, key=lambda c: len(c["prompt"]))
    cases.append({"name": "merged-multitenant", "kind": "merged", "model": "union",
                  "specials": allspec, "prompt": longest["prompt"]})
    print(f"OK  merged specials={len(allspec)} (prompt borrowed from {longest['name']})")

json.dump(cases, open(OUT, "w"))
print("WROTE", OUT, "cases=", len(cases))
