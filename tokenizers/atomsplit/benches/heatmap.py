#!/usr/bin/env python3
"""Heatmap of the atomsplit `regex` bench: our classify+fsm pipeline speedup vs each SOTA engine,
per pre-tokenizer × language. Green = we win big, red = close race / behind."""
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

LOG = sys.argv[1]
OUT = sys.argv[2]

ENGINES = ["gpt2", "cl100k", "o200k", "deepseek", "ws_split", "whitespace", "digits", "punct", "bert"]
REFS = ["onig", "fancy", "logos", "pcre2"]
# discover languages in first-seen order
langs = []
# data[pretok][lang] = [vsOnig, vsFncy, vsLogos, vsPcre2]
data = {e: {} for e in ENGINES}

for line in open(LOG):
    toks = [t for t in line.split() if t != "|"]
    if len(toks) < 15 or toks[0] not in data:
        continue
    pre, lang = toks[0], toks[1]
    # ...clsSIMD clsScal fsm onig fancy logos pcre2 | vsOnig vsFncy vsLogos vsPcre2 ok
    vs = toks[11:15]
    def num(x):
        x = x.rstrip("x")
        return np.nan if x in ("—", "-", "") else float(x)
    data[pre][lang] = [num(v) for v in vs]
    if lang not in langs:
        langs.append(lang)

cmap = plt.get_cmap("RdYlGn").copy()
cmap.set_bad("#d9d9d9")
norm = LogNorm(vmin=0.8, vmax=60)

fig, axes = plt.subplots(3, 3, figsize=(15.5, 15), constrained_layout=True)
for ax, pre in zip(axes.flat, ENGINES):
    M = np.array([data[pre].get(l, [np.nan] * 4) for l in langs])  # [langs x refs]
    ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(pre, fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(REFS)))
    ax.set_xticklabels(REFS, fontsize=9)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs, fontsize=8)
    ax.tick_params(length=0)
    for i in range(len(langs)):
        for j in range(len(REFS)):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#666")
            else:
                # white text on the dark-red/dark-green extremes, black in the middle
                t = norm(v)
                col = "white" if (t < 0.18 or t > 0.86) else "black"
                ax.text(j, i, f"{v:.0f}×" if v >= 10 else f"{v:.1f}×",
                        ha="center", va="center", fontsize=7.5, color=col, fontweight="bold")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=axes, shrink=0.5, aspect=30, pad=0.02,
                  ticks=[1, 2, 5, 10, 20, 50])
cb.ax.set_yticklabels(["1× (tie)", "2×", "5×", "10×", "20×", "50×"])
cb.set_label("speedup = engine ÷ our pipeline (SIMD classify + scalar fsm)  ·  green = we win big",
             fontsize=10)

fig.suptitle("atomsplit pre-tokenization: our classify+fsm pipeline vs SOTA splitters\n"
             "(aarch64 / Apple Silicon, release; onig & pcre2-JIT = C, fancy = pure-Rust regex, "
             "logos = compile-time DFA lexer; n/a = engine can't express that split)",
             fontsize=13, fontweight="bold")
fig.savefig(OUT, format="svg", bbox_inches="tight")
print(f"wrote {OUT}: {len(langs)} langs × {len(REFS)} engines × {len(ENGINES)} pre-tokenizers")
