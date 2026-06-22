"""Real-world tokenizer throughput benchmark.

Compares up to five backends head-to-head on a standardized
``(batch_size, input_length_tokens)`` matrix — the same knobs fastokens' and
wordchipper's ablation scripts sweep over:

    * ``tokenizers``                (this repo)
    * ``tiktoken``                  (OpenAI)
    * ``wordchipper``               (https://github.com/zspacelabs/wordchipper)
    * ``iree.tokenizer``            (https://github.com/iree-org/iree-tokenizer-py)
    * ``bpe`` / ``bpe-openai``      (https://github.com/github/rust-gems/tree/main/crates/bpe)

Real prompts are sourced from ``zai-org/LongBench-v2`` (same dataset used by
fastokens' ``simple_bench.rs``), tokenized once, then truncated/repeated per
prompt to reach exactly ``input_length`` tokens — mirroring fastokens'
``_adjust_tokens`` helper and wordchipper's fineweb batch approach.

Both **encode** and **decode** are benchmarked over the same matrix. Results
stream live into colored ``rich`` tables.

Backends that are not installed are gracefully skipped.
"""

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import time
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Backends whose encode outputs must agree for a given encoding.
ENCODINGS: Dict[str, Dict[str, str]] = {
    "cl100k_base": {"hf_repo": "Xenova/text-embedding-ada-002"},
    "o200k_base":  {"hf_repo": "Xenova/gpt-4o"},
    "gpt2":        {"hf_repo": "Xenova/gpt2"},
    "llama3":      {"hf_repo": "meta-llama/Llama-3.2-1B"},
}
DEFAULT_ENCODING = "cl100k_base"
DATASET_REPO = "zai-org/LongBench-v2"
DATASET_FILE = "data.json"

# Matrix follows fastokens' ``ablation.sh`` (batches) and the token-length
# buckets from ``dynamo_speed.py``.
DEFAULT_BATCH_SIZES = [1, 8, 32, 128, 512]
DEFAULT_INPUT_LENGTHS = [128, 512, 2048, 8192, 32768]
ALL_BACKENDS = ["tokenizers", "tiktoken", "wordchipper", "iree", "bpe"]

LLAMA_PAT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------


class Backend:
    """Uniform encode/decode API across the four tokenizers under test."""

    def __init__(self, name: str, encode_batch: Callable, decode_batch: Callable) -> None:
        self.name = name
        self.encode_batch = encode_batch
        self.decode_batch = decode_batch


def _resolve_hf_repo(model: str) -> str:
    """Map a short encoding name to an HF repo, or pass through as-is."""
    if model in ENCODINGS:
        return ENCODINGS[model]["hf_repo"]
    return model


def _load_hf(model: str, num_threads: int) -> Optional[Backend]:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    os.environ["RAYON_NUM_THREADS"] = str(num_threads)
    try:
        tok = Tokenizer.from_pretrained(_resolve_hf_repo(model))
    except Exception:
        return None

    def encode_batch(texts: List[str]) -> List[List[int]]:
        out = tok.encode_batch_fast(texts, add_special_tokens=False)
        return [e.ids for e in out]

    def decode_batch(ids_list: List[List[int]]) -> List[str]:
        return tok.decode_batch(ids_list, skip_special_tokens=False)

    return Backend("tokenizers", encode_batch, decode_batch)


def _load_tiktoken(model: str, num_threads: int) -> Optional[Backend]:
    try:
        import tiktoken
    except ImportError:
        return None

    if model in ("cl100k_base", "o200k_base", "gpt2", "r50k_base", "p50k_base"):
        enc = tiktoken.get_encoding(model)
    else:
        # Try to load a tiktoken-format ``original/tokenizer.model`` from the
        # HF repo. This only works for llama-3-style BPE models.
        from tiktoken.load import load_tiktoken_bpe
        repo = _resolve_hf_repo(model)
        try:
            path = hf_hub_download(repo, "original/tokenizer.model")
        except Exception:
            return None
        try:
            ranks = load_tiktoken_bpe(path)
        except Exception:
            return None
        specials = [
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        ]
        specials_map = {t: len(ranks) + i for i, t in enumerate(specials)}
        enc = tiktoken.Encoding(
            name=model.replace("/", "_"),
            pat_str=LLAMA_PAT,
            mergeable_ranks=ranks,
            special_tokens=specials_map,
        )

    def encode_batch(texts: List[str]) -> List[List[int]]:
        return enc.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode_batch(ids_list: List[List[int]]) -> List[str]:
        return enc.decode_batch(ids_list, num_threads=num_threads)

    return Backend("tiktoken", encode_batch, decode_batch)


def _load_wordchipper(model: str, num_threads: int) -> Optional[Backend]:
    try:
        import wordchipper as wc
    except ImportError:
        return None
    # wordchipper only ships OpenAI encodings.
    if model not in ("gpt2", "cl100k_base", "r50k_base", "p50k_base", "o200k_base"):
        return None

    options = wc.TokenizerOptions.default()
    if hasattr(options, "set_parallel"):
        options.set_parallel(num_threads > 1)
    if hasattr(options, "set_accelerated_lexers"):
        options.set_accelerated_lexers(True)
    tok = wc.Tokenizer.from_pretrained(model, options)

    def encode_batch(texts: List[str]) -> List[List[int]]:
        return tok.encode_batch(texts)

    def decode_batch(ids_list: List[List[int]]) -> List[str]:
        return tok.decode_batch(ids_list)

    return Backend("wordchipper", encode_batch, decode_batch)


def _load_iree(model: str, num_threads: int) -> Optional[Backend]:
    try:
        from iree.tokenizer import Tokenizer as IreeTokenizer
    except ImportError:
        return None
    try:
        path = hf_hub_download(_resolve_hf_repo(model), "tokenizer.json")
    except Exception:
        return None
    try:
        tok = IreeTokenizer.from_file(path)
    except Exception:
        return None

    def encode_batch(texts: List[str]) -> List[List[int]]:
        return tok.encode_batch(texts)

    def decode_batch(ids_list: List[List[int]]) -> List[str]:
        return tok.decode_batch(ids_list)

    return Backend("iree", encode_batch, decode_batch)


def _load_bpe(model: str, num_threads: int) -> Optional[Backend]:
    """github/rust-gems ``bpe-openai`` crate (Python wheel ``bpe-openai``).

    Only ships OpenAI-compatible encodings. Returns ``None`` for arbitrary HF
    repos, matching wordchipper's scope.
    """
    try:
        import bpe_openai
    except ImportError:
        return None
    try:
        names = set(bpe_openai.list_encoding_names())
    except Exception:
        return None
    if model not in names:
        return None
    try:
        enc = bpe_openai.get_encoding(model)
    except Exception:
        return None

    def encode_batch(texts: List[str]) -> List[List[int]]:
        return enc.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode_batch(ids_list: List[List[int]]) -> List[str]:
        return enc.decode_batch(ids_list, num_threads=num_threads)

    return Backend("bpe", encode_batch, decode_batch)


BACKEND_LOADERS: Dict[str, Callable[[str, int], Optional[Backend]]] = {
    "tokenizers": _load_hf,
    "tiktoken": _load_tiktoken,
    "wordchipper": _load_wordchipper,
    "iree": _load_iree,
    "bpe": _load_bpe,
}


# ---------------------------------------------------------------------------
# Data / corpus
# ---------------------------------------------------------------------------


def _load_prompts(num_prompts: int) -> List[str]:
    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    out: List[str] = []
    for item in data:
        ctx = (item.get("context") or "").strip()
        if ctx:
            out.append(ctx)
        if len(out) >= num_prompts:
            break
    return out


def _adjust_tokens(ids: List[int], target: int) -> List[int]:
    if len(ids) >= target:
        return ids[:target]
    reps = (target // len(ids)) + 1
    return (ids * reps)[:target]


def _build_sample_pool(
    reference: Backend, prompts: List[str], input_lengths: List[int],
) -> Dict[int, List[str]]:
    """Pre-encode prompts once, then derive per-length texts.

    For each target length ``L`` we produce ``len(prompts)`` strings; at bench
    time we slice the first ``batch_size`` of them.
    """
    ids_pool = [reference.encode_batch([p])[0] for p in prompts]
    ids_pool = [ids for ids in ids_pool if ids]
    # Decode adjusted ids back to text using the *reference* backend, so the
    # text round-trips exactly to the target length for every backend that
    # agrees with the reference encoding.
    pool: Dict[int, List[str]] = {}
    for L in input_lengths:
        adjusted = [_adjust_tokens(ids, L) for ids in ids_pool]
        decoded = reference.decode_batch(adjusted)
        pool[L] = decoded
    return pool


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time(fn: Callable, warmup: int = 1, iters: int = 3) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        dt = time.perf_counter_ns() - t0
        if dt < best:
            best = dt
    return best / 1e9


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _worker(
    model: str,
    backends_wanted: List[str],
    combos: List[Tuple[int, int]],
    num_prompts: int,
    num_threads: int,
    iters: int,
    warmup: int,
    q: "Queue",
) -> None:
    try:
        # Order matters: we prefer to use a reference backend for sample
        # generation. tokenizers > tiktoken > wordchipper > iree.
        loaded: Dict[str, Backend] = {}
        for name in backends_wanted:
            loader = BACKEND_LOADERS[name]
            try:
                b = loader(model, num_threads)
            except Exception as e:  # noqa: BLE001
                q.put({"log": f"[yellow]skip[/yellow] {name}: {e!r}"})
                b = None
            if b is None:
                q.put({"log": f"[yellow]skip[/yellow] {name} (not available for {model})"})
            else:
                loaded[name] = b
                q.put({"log": f"[green]ok[/green] {name} loaded"})

        if not loaded:
            q.put({"error": "no backend could be loaded"})
            return

        reference = next(iter(loaded.values()))
        q.put({"log": f"[dim]reference backend for sample generation: {reference.name}[/dim]"})

        # Cross-backend correctness on a canonical short input.
        probe = "Hello world, this is a test of tokenizer equivalence."
        ref_ids = reference.encode_batch([probe])[0]
        for name, b in loaded.items():
            if b is reference:
                continue
            ids = b.encode_batch([probe])[0]
            if ids != ref_ids:
                q.put({"log": f"[red]mismatch[/red] {name} vs {reference.name} on probe"})
            else:
                q.put({"log": f"[dim]agree[/dim] {name} == {reference.name}"})

        prompts = _load_prompts(num_prompts)
        input_lengths = sorted({L for _, L in combos})
        text_pool = _build_sample_pool(reference, prompts, input_lengths)

        for batch_size, input_length in combos:
            texts_all = text_pool[input_length]
            texts = [texts_all[i % len(texts_all)] for i in range(batch_size)]
            total_bytes = sum(len(t.encode("utf-8")) for t in texts)

            # Encode
            for name, b in loaded.items():
                sec = _time(lambda b=b: b.encode_batch(texts), warmup=warmup, iters=iters)
                q.put({
                    "phase": "encode",
                    "backend": name,
                    "batch_size": batch_size,
                    "input_length": input_length,
                    "total_bytes": total_bytes,
                    "sec": sec,
                })

            # Decode: pre-encode with each backend so every backend decodes its
            # own ids (avoids cross-backend id-space drift on llama3).
            for name, b in loaded.items():
                ids_list = b.encode_batch(texts)
                tok_count = sum(len(ids) for ids in ids_list)
                sec = _time(lambda b=b, ids_list=ids_list: b.decode_batch(ids_list),
                            warmup=warmup, iters=iters)
                q.put({
                    "phase": "decode",
                    "backend": name,
                    "batch_size": batch_size,
                    "input_length": input_length,
                    "total_bytes": total_bytes,
                    "total_tokens": tok_count,
                    "sec": sec,
                })
    except Exception as e:  # noqa: BLE001
        q.put({"error": repr(e)})
    finally:
        q.put(None)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


BACKEND_STYLE = {
    "tokenizers": "bold green",
    "tiktoken":   "bold yellow",
    "wordchipper": "bold magenta",
    "iree":       "bold cyan",
    "bpe":        "bold blue",
}


def _color_speedup(ratio: float) -> str:
    if ratio >= 1.10:
        return f"[bold green]{ratio:.2f}×[/bold green]"
    if ratio <= 0.90:
        return f"[bold red]{ratio:.2f}×[/bold red]"
    return f"[white]{ratio:.2f}×[/white]"


def _build_matrix_table(
    phase: str, rows: List[dict], backend_order: List[str], throughput_unit: str,
) -> Table:
    """One row per (batch_size, input_length); one ms column and one throughput column per backend."""
    grouped: Dict[Tuple[int, int], Dict[str, dict]] = {}
    for r in rows:
        if r["phase"] != phase:
            continue
        grouped.setdefault((r["batch_size"], r["input_length"]), {})[r["backend"]] = r

    t = Table(
        title=f"[bold]{phase} throughput — {throughput_unit}[/bold]",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_justify="left",
        expand=False,
    )
    t.add_column("batch", justify="right", style="cyan", no_wrap=True)
    t.add_column("input_len", justify="right", style="cyan", no_wrap=True)
    for name in backend_order:
        style = BACKEND_STYLE.get(name, "white")
        t.add_column(f"{name}\nms", justify="right", style=style, no_wrap=True)
        t.add_column(f"{name}\n{throughput_unit}", justify="right", style=style, no_wrap=True)
    t.add_column("winner", justify="center", no_wrap=True)

    last_bs = None
    for (bs, L) in sorted(grouped.keys()):
        cell = grouped[(bs, L)]
        if last_bs is not None and bs != last_bs:
            t.add_section()
        last_bs = bs

        row: List[str] = [str(bs), str(L)]
        throughputs: Dict[str, float] = {}
        for name in backend_order:
            r = cell.get(name)
            if r is None:
                row.extend(["-", "-"])
                continue
            ms = r["sec"] * 1000
            if phase == "encode":
                through = r["total_bytes"] / r["sec"] / 1e6  # MB/s
            else:
                through = r.get("total_tokens", 0) / r["sec"] / 1e6  # Mtok/s
            throughputs[name] = through
            row.append(f"{ms:,.2f}")
            row.append(f"{through:,.1f}")

        if throughputs:
            winner = max(throughputs, key=throughputs.get)
            style = BACKEND_STYLE.get(winner, "white")
            row.append(f"[{style}]{winner}[/{style}]")
        else:
            row.append("-")
        t.add_row(*row)
    return t


def _summary_panel(rows: List[dict], backend_order: List[str], phase: str) -> Panel:
    """Geo-mean speedup of tokenizers over each competitor for the given phase."""
    if "tokenizers" not in backend_order:
        return Panel("[dim]tokenizers backend missing — no relative summary[/dim]", box=box.ROUNDED)

    per_combo: Dict[Tuple[int, int], Dict[str, float]] = {}
    for r in rows:
        if r["phase"] != phase:
            continue
        per_combo.setdefault((r["batch_size"], r["input_length"]), {})[r["backend"]] = r["sec"]

    parts: List[str] = []
    for other in backend_order:
        if other == "tokenizers":
            continue
        ratios = []
        for combo, backends in per_combo.items():
            if "tokenizers" in backends and other in backends:
                if backends["tokenizers"] > 0 and backends[other] > 0:
                    ratios.append(backends[other] / backends["tokenizers"])
        if not ratios:
            continue
        gmean = statistics.geometric_mean(ratios)
        best = max(ratios)
        worst = min(ratios)
        parts.append(
            f"vs [{BACKEND_STYLE.get(other, 'white')}]{other}[/]: "
            f"geo-mean {_color_speedup(gmean)}  "
            f"range [{worst:.2f}× .. {best:.2f}×]"
        )
    body = "\n".join(parts) if parts else "[dim]no pairwise data[/dim]"
    return Panel(body, title=f"{phase} summary (higher is better for tokenizers)", box=box.ROUNDED)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    model: str,
    backends_wanted: List[str],
    batch_sizes: List[int],
    input_lengths: List[int],
    num_threads: int,
    num_prompts: int,
    iters: int,
    warmup: int,
    console: Optional[Console] = None,
    print_tables: bool = True,
) -> Tuple[List[dict], List[str]]:
    """Run the (batch × input_length) matrix for a single model.

    Returns ``(rows, backend_order)`` so callers (e.g., multi-model suites)
    can aggregate without re-parsing printed output.
    """
    console = console or Console()
    combos = [(b, L) for b in batch_sizes for L in input_lengths]

    console.print(Panel(
        f"[bold]model:[/bold] [cyan]{model}[/cyan]   "
        f"[bold]backends:[/bold] {backends_wanted}\n"
        f"[bold]batch sizes:[/bold]   {batch_sizes}\n"
        f"[bold]input lengths:[/bold] {input_lengths}\n"
        f"[bold]threads:[/bold] {num_threads}   "
        f"[bold]iters:[/bold] {iters} (warmup {warmup})   "
        f"[bold]combos:[/bold] {len(combos)}",
        title="benchmark configuration", box=box.ROUNDED, title_align="left",
    ))

    q: "Queue" = Queue()
    p = Process(
        target=_worker,
        args=(model, backends_wanted, combos, num_prompts,
              num_threads, iters, warmup, q),
    )
    p.start()

    rows: List[dict] = []
    error: Optional[str] = None
    ordered: List[str] = []
    expected = len(backends_wanted) * len(combos) * 2

    def _status(done: int, current: Optional[dict]) -> Panel:
        bar = "█" * int(40 * done / max(expected, 1))
        bar = bar.ljust(40, "·")
        line = f"[cyan]{bar}[/cyan]  {done}/{expected}"
        if current is not None:
            line += (
                f"   [dim]{current['phase']} "
                f"{current['backend']} bs={current['batch_size']} "
                f"len={current['input_length']} "
                f"{current['sec']*1000:.2f}ms[/dim]"
            )
        return Panel(line, title="benchmark progress", box=box.ROUNDED, title_align="left")

    last_item: Optional[dict] = None
    with Live(_status(0, None), console=console, refresh_per_second=8) as live:
        while True:
            item = q.get()
            if item is None:
                break
            if "log" in item:
                live.console.log(item["log"])
                continue
            if "error" in item:
                error = item["error"]
                continue
            if item["backend"] not in ordered:
                ordered.append(item["backend"])
            rows.append(item)
            last_item = item
            live.update(_status(len(rows), last_item))

    p.join()

    if print_tables:
        console.print()
        console.print(_build_matrix_table("encode", rows, ordered, "MB/s"))
        console.print(_summary_panel(rows, ordered, "encode"))
        console.print()
        console.print(_build_matrix_table("decode", rows, ordered, "Mtok/s"))
        console.print(_summary_panel(rows, ordered, "decode"))

    if error:
        console.print(f"[bold red]worker error for {model}:[/bold red] {error}")

    return rows, ordered


# ---------------------------------------------------------------------------
# Multi-model suite
# ---------------------------------------------------------------------------


def _best_throughput(
    rows: List[dict], backend: str, phase: str, kind: str,
) -> Optional[float]:
    """Return the peak throughput (MB/s for encode, Mtok/s for decode) over all combos."""
    best = None
    for r in rows:
        if r["backend"] != backend or r["phase"] != phase:
            continue
        if r["sec"] <= 0:
            continue
        if kind == "MB/s":
            v = r["total_bytes"] / r["sec"] / 1e6
        else:
            v = r.get("total_tokens", 0) / r["sec"] / 1e6
        if best is None or v > best:
            best = v
    return best


def _build_leaderboard(
    per_model: Dict[str, Tuple[List[dict], List[str]]],
    phase: str,
    unit: str,
) -> Table:
    """Cross-model leaderboard: one row per model, one column per backend."""
    backends_seen: List[str] = []
    for _, order in per_model.values():
        for b in order:
            if b not in backends_seen:
                backends_seen.append(b)

    t = Table(
        title=f"[bold]{phase} peak throughput per model ({unit})[/bold]",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_justify="left",
        expand=False,
    )
    t.add_column("model", style="cyan", no_wrap=True)
    for name in backends_seen:
        style = BACKEND_STYLE.get(name, "white")
        t.add_column(name, justify="right", style=style, no_wrap=True)
    if "tokenizers" in backends_seen and len(backends_seen) > 1:
        t.add_column("best ×\ntokenizers", justify="right", no_wrap=True)

    for model, (rows, _order) in per_model.items():
        vals: Dict[str, Optional[float]] = {
            b: _best_throughput(rows, b, phase, unit)
            for b in backends_seen
        }
        row = [model]
        for b in backends_seen:
            v = vals[b]
            row.append(f"{v:,.1f}" if v is not None else "-")
        if "tokenizers" in backends_seen and len(backends_seen) > 1:
            tk = vals.get("tokenizers") or 0.0
            best_other_name, best_other_val = None, 0.0
            for b in backends_seen:
                if b == "tokenizers":
                    continue
                v = vals.get(b)
                if v is not None and v > best_other_val:
                    best_other_val, best_other_name = v, b
            if tk > 0 and best_other_val > 0 and best_other_name:
                ratio = best_other_val / tk
                style = BACKEND_STYLE.get(best_other_name, "white")
                marker = _color_speedup(ratio)
                row.append(f"{marker} [{style}]{best_other_name}[/{style}]")
            else:
                row.append("-")
        t.add_row(*row)
    return t


def _machine_state() -> Dict[str, Any]:
    """Snapshot of machine state for fairness reporting."""
    info: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nproc": os.cpu_count(),
    }
    try:
        with open("/proc/loadavg") as f:
            info["loadavg"] = f.read().split()[:3]
    except OSError:
        info["loadavg"] = None
    try:
        info["pinned_cpus"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        info["pinned_cpus"] = None
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            info["cpu_governor"] = f.read().strip()
    except OSError:
        info["cpu_governor"] = None
    try:
        out = subprocess.check_output(
            ["lscpu"], stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        for line in out.splitlines():
            if line.startswith("Model name:"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except (subprocess.SubprocessError, FileNotFoundError):
        info["cpu_model"] = None
    return info


def _preflight(console: Console, fail_on_load: bool = False) -> Dict[str, Any]:
    """Print a fairness preflight; optionally abort on heavy load."""
    state = _machine_state()
    loadavg = state.get("loadavg") or ["?", "?", "?"]
    pinned = state.get("pinned_cpus")
    pinned_str = (
        f"{len(pinned)} ({pinned[0]}..{pinned[-1]})"
        if pinned and len(pinned) > 4
        else str(pinned)
    )
    lines = [
        f"[bold]host:[/bold] {state.get('cpu_model') or 'unknown CPU'}   "
        f"[bold]nproc:[/bold] {state.get('nproc')}",
        f"[bold]load avg (1/5/15):[/bold] {loadavg[0]} / {loadavg[1]} / {loadavg[2]}",
        f"[bold]pinned cpus:[/bold] {pinned_str}",
        f"[bold]governor:[/bold] {state.get('cpu_governor') or 'unknown'}",
    ]
    console.print(Panel("\n".join(lines), title="fairness preflight", box=box.ROUNDED, title_align="left"))

    nproc = state.get("nproc") or 1
    try:
        one_min = float(loadavg[0])
    except (TypeError, ValueError):
        one_min = 0.0
    # Warn if load > 50% of cores; abort only if --strict requested.
    if one_min > nproc * 0.5:
        msg = f"[bold red]high system load ({one_min} over {nproc} cores)[/bold red]"
        console.print(msg)
        if fail_on_load:
            raise SystemExit(2)
    return state


def _serialize(per_model: Dict[str, Tuple[List[dict], List[str]]],
               state: Dict[str, Any],
               config: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "state": state,
        "config": config,
        "models": {},
    }
    for model, (rows, order) in per_model.items():
        peaks: Dict[str, Dict[str, float]] = {}
        for b in order:
            peaks[b] = {
                "encode_mb_s": _best_throughput(rows, b, "encode", "MB/s") or 0.0,
                "decode_mtok_s": _best_throughput(rows, b, "decode", "Mtok/s") or 0.0,
            }
        out["models"][model] = {"backends": order, "peaks": peaks, "rows": rows}
    return out


def _markdown_leaderboard(per_model: Dict[str, Tuple[List[dict], List[str]]],
                          state: Dict[str, Any],
                          config: Dict[str, Any]) -> str:
    backends_seen: List[str] = []
    for _, order in per_model.values():
        for b in order:
            if b not in backends_seen:
                backends_seen.append(b)

    parts: List[str] = []
    parts.append(f"# Tokenizer benchmark results\n")
    parts.append(f"- **Timestamp**: {state.get('timestamp')}")
    parts.append(f"- **CPU**: {state.get('cpu_model') or 'unknown'}   (nproc: {state.get('nproc')})")
    parts.append(f"- **Load avg (1/5/15)**: {' / '.join(map(str, state.get('loadavg') or []))}")
    pinned = state.get('pinned_cpus') or []
    parts.append(f"- **Pinned CPUs**: {len(pinned)} ({min(pinned)}..{max(pinned)})"
                 if pinned else "- **Pinned CPUs**: n/a")
    parts.append(f"- **Governor**: {state.get('cpu_governor') or 'unknown'}")
    parts.append(f"- **Config**: batches={config['batch_sizes']} lengths={config['input_lengths']} "
                 f"threads={config['num_threads']} iters={config['iters']} warmup={config['warmup']}")
    parts.append("")

    for phase, unit in (("encode", "MB/s"), ("decode", "Mtok/s")):
        parts.append(f"## {phase} peak throughput ({unit})\n")
        header = ["model"] + backends_seen + (["best × tokenizers"] if "tokenizers" in backends_seen else [])
        parts.append("| " + " | ".join(header) + " |")
        parts.append("|" + "|".join(["---"] * len(header)) + "|")
        for model, (rows, _order) in per_model.items():
            cells = [model]
            vals: Dict[str, Optional[float]] = {}
            for b in backends_seen:
                v = _best_throughput(rows, b, phase, unit)
                vals[b] = v
                cells.append(f"{v:,.1f}" if v is not None else "–")
            if "tokenizers" in backends_seen:
                tk = vals.get("tokenizers") or 0.0
                best_name, best_val = None, 0.0
                for b in backends_seen:
                    if b == "tokenizers":
                        continue
                    v = vals.get(b)
                    if v is not None and v > best_val:
                        best_val, best_name = v, b
                if tk > 0 and best_val > 0 and best_name:
                    cells.append(f"{best_val/tk:.2f}× {best_name}")
                else:
                    cells.append("–")
            parts.append("| " + " | ".join(cells) + " |")
        parts.append("")
    return "\n".join(parts)


def run_suite(
    models: List[str],
    backends_wanted: List[str],
    batch_sizes: List[int],
    input_lengths: List[int],
    num_threads: int,
    num_prompts: int,
    iters: int,
    warmup: int,
    save_json: Optional[str] = None,
    save_md: Optional[str] = None,
    strict_fairness: bool = False,
) -> None:
    console = Console()
    state = _preflight(console, fail_on_load=strict_fairness)

    per_model: Dict[str, Tuple[List[dict], List[str]]] = {}

    for idx, model in enumerate(models, 1):
        console.rule(f"[bold cyan]{idx}/{len(models)}  {model}[/bold cyan]")
        # 2-second settle between runs to let thermals and caches reset.
        if idx > 1:
            time.sleep(2)
        rows, order = run(
            model=model,
            backends_wanted=backends_wanted,
            batch_sizes=batch_sizes,
            input_lengths=input_lengths,
            num_threads=num_threads,
            num_prompts=num_prompts,
            iters=iters,
            warmup=warmup,
            console=console,
            print_tables=True,
        )
        per_model[model] = (rows, order)

    console.rule("[bold]cross-model leaderboard[/bold]")
    console.print()
    console.print(_build_leaderboard(per_model, "encode", "MB/s"))
    console.print()
    console.print(_build_leaderboard(per_model, "decode", "Mtok/s"))

    config = dict(
        batch_sizes=batch_sizes,
        input_lengths=input_lengths,
        num_threads=num_threads,
        num_prompts=num_prompts,
        iters=iters,
        warmup=warmup,
        backends=backends_wanted,
    )

    if save_json:
        payload = _serialize(per_model, state, config)
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        console.print(f"[dim]wrote {save_json}[/dim]")
    if save_md:
        md = _markdown_leaderboard(per_model, state, config)
        os.makedirs(os.path.dirname(save_md) or ".", exist_ok=True)
        with open(save_md, "w") as f:
            f.write(md)
        console.print(f"[dim]wrote {save_md}[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# Curated non-OpenAI BPE models for the default --hf-models sweep.
DEFAULT_HF_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-V3",
    "zai-org/GLM-4.5-Air",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "01-ai/Yi-1.5-9B",
    "bigcode/starcoder2-7b",
    "EleutherAI/gpt-neox-20b",
    "tiiuae/falcon-7b",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bench_tokenizer",
        description="Cross-library tokenizer benchmark "
                    "(tokenizers vs tiktoken vs wordchipper vs iree).",
    )
    parser.add_argument(
        "-e", "--encoding", default=None,
        help="Run a single OpenAI encoding (gpt2/cl100k_base/o200k_base/llama3) "
             "or any HF repo id.",
    )
    parser.add_argument(
        "--hf-models", nargs="*", default=None,
        help="Run the matrix for multiple HF repo ids and print a leaderboard. "
             f"If given with no args, uses the default list: {DEFAULT_HF_MODELS}.",
    )
    parser.add_argument(
        "--backends", nargs="+", default=ALL_BACKENDS, choices=ALL_BACKENDS,
    )
    parser.add_argument(
        "-b", "--batch-sizes", nargs="+", default=DEFAULT_BATCH_SIZES, type=int,
    )
    parser.add_argument(
        "-l", "--input-lengths", nargs="+", default=DEFAULT_INPUT_LENGTHS, type=int,
    )
    parser.add_argument("-t", "--threads", default=os.cpu_count() or 8, type=int)
    parser.add_argument("-p", "--num-prompts", default=16, type=int)
    parser.add_argument("--iters", default=3, type=int)
    parser.add_argument("--warmup", default=1, type=int)
    parser.add_argument("--save-json", default=None, type=str,
                        help="Write full results (rows + peaks + machine state) to this path.")
    parser.add_argument("--save-md", default=None, type=str,
                        help="Write a markdown leaderboard to this path.")
    parser.add_argument("--strict-fairness", action="store_true",
                        help="Abort when system load exceeds 50% of nproc.")
    args = parser.parse_args()

    if args.hf_models is not None:
        models = args.hf_models or DEFAULT_HF_MODELS
        run_suite(
            models=models,
            backends_wanted=args.backends,
            batch_sizes=args.batch_sizes,
            input_lengths=args.input_lengths,
            num_threads=args.threads,
            num_prompts=args.num_prompts,
            iters=args.iters,
            warmup=args.warmup,
            save_json=args.save_json,
            save_md=args.save_md,
            strict_fairness=args.strict_fairness,
        )
    else:
        console = Console()
        state = _preflight(console, fail_on_load=args.strict_fairness)
        rows, order = run(
            model=args.encoding or DEFAULT_ENCODING,
            backends_wanted=args.backends,
            batch_sizes=args.batch_sizes,
            input_lengths=args.input_lengths,
            num_threads=args.threads,
            num_prompts=args.num_prompts,
            iters=args.iters,
            warmup=args.warmup,
        )
        if args.save_json or args.save_md:
            per_model = {args.encoding or DEFAULT_ENCODING: (rows, order)}
            config = dict(
                batch_sizes=args.batch_sizes,
                input_lengths=args.input_lengths,
                num_threads=args.threads,
                num_prompts=args.num_prompts,
                iters=args.iters,
                warmup=args.warmup,
                backends=args.backends,
            )
            if args.save_json:
                payload = _serialize(per_model, state, config)
                os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
                with open(args.save_json, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
            if args.save_md:
                md = _markdown_leaderboard(per_model, state, config)
                os.makedirs(os.path.dirname(args.save_md) or ".", exist_ok=True)
                with open(args.save_md, "w") as f:
                    f.write(md)


if __name__ == "__main__":
    main()
