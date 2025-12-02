#!/usr/bin/env python3
"""
Benchmark automation script for tokenizer variants
Runs each variant 3 times and generates a TSV report with statistics
"""

import subprocess
import time
import sys
import os
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any
import json

SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent
BENCHMARKS_DIR = SCRIPT_DIR

# Configuration
NUM_RUNS = 3
INPUT_FILE = BENCHMARKS_DIR / "big.txt"
TOKENIZER_FILE = ROOT_DIR / "tokenizers" / "data" / "tokenizer.json"

# Variant configurations
VARIANTS = {
    "tokenizers-rust": {
        "command": [str(BENCHMARKS_DIR / "bench_rust.out"), str(TOKENIZER_FILE), str(INPUT_FILE)],
        "name": "Rust"
    },
    "tokenizers-python": {
        "command": ["python3", str(BENCHMARKS_DIR / "bench_python.py"), str(TOKENIZER_FILE), str(INPUT_FILE)],
        "name": "Python"
    },
    "tokenizers-c": {
        "command": [str(BENCHMARKS_DIR / "bench_c.out"), str(TOKENIZER_FILE), str(INPUT_FILE)],
        "name": "C Bindings",
        "env": {"LD_LIBRARY_PATH": str(ROOT_DIR / "bindings/c/target/release")}
    },
    "tokenizers-cpp-bindings": {
        "command": [str(BENCHMARKS_DIR / "bench_cpp_bindings.out"), str(TOKENIZER_FILE), str(INPUT_FILE)],
        "name": "C++ Bindings",
        "env": {"LD_LIBRARY_PATH": str(ROOT_DIR / "bindings/c/target/release")}
    }
}


def parse_output(output: str) -> Dict[str, float]:
    """Parse the benchmark output into a dictionary"""
    result = {}
    for line in output.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def run_benchmark(variant_key: str, config: Dict[str, Any]) -> Dict[str, float]:
    """Run a single benchmark and return the parsed results"""
    env = os.environ.copy()
    if "env" in config:
        env.update(config["env"])
    
    try:
        result = subprocess.run(
            config["command"],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        return parse_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {variant_key}:", file=sys.stderr)
        print(f"Command: {' '.join(config['command'])}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print(f"Stdout: {e.stdout}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError as e:
        print(f"Error: Could not find executable for {variant_key}", file=sys.stderr)
        print(f"Command: {' '.join(config['command'])}", file=sys.stderr)
        print(f"Make sure to run build.sh first", file=sys.stderr)
        raise


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate mean and standard deviation"""
    if len(values) < 2:
        return {"mean": values[0] if values else 0, "stdev": 0}
    return {"mean": mean(values), "stdev": stdev(values)}


def main():
    print("=== Tokenizer Benchmark Suite ===")
    print(f"Input file: {INPUT_FILE}")
    print(f"Tokenizer: {TOKENIZER_FILE}")
    print(f"Number of runs per variant: {NUM_RUNS}")
    print()
    
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)
    
    if not TOKENIZER_FILE.exists():
        print(f"Error: Tokenizer file not found: {TOKENIZER_FILE}", file=sys.stderr)
        sys.exit(1)
    
    all_results = {}
    
    for variant_key, config in VARIANTS.items():
        variant_name = config["name"]
        print(f">>> Running {variant_name} ({NUM_RUNS} runs)...")
        
        runs = []
        for run_num in range(1, NUM_RUNS + 1):
            print(f"    Run {run_num}/{NUM_RUNS}...", end=" ", flush=True)
            try:
                result = run_benchmark(variant_key, config)
                runs.append(result)
                print(f"✓ ({result.get('encode_time_ms', 0):.0f}ms)")
            except Exception as e:
                print(f"✗ FAILED")
                print(f"    Error: {e}", file=sys.stderr)
                # Store None to indicate failure
                all_results[variant_key] = None
                break
        else:
            # All runs succeeded
            all_results[variant_key] = {
                "name": variant_name,
                "runs": runs
            }
        
        print()
    
    # Generate statistics
    print("=== Calculating Statistics ===")
    print()
    
    stats = {}
    for variant_key, data in all_results.items():
        if data is None:
            print(f"{VARIANTS[variant_key]['name']}: FAILED")
            continue
        
        load_times = [r['load_time_ms'] for r in data['runs']]
        encode_times = [r['encode_time_ms'] for r in data['runs']]
        tokens_per_sec = [r['tokens_per_sec'] for r in data['runs']]
        
        stats[variant_key] = {
            "name": data["name"],
            "load_time": calculate_stats(load_times),
            "encode_time": calculate_stats(encode_times),
            "tokens_per_sec": calculate_stats(tokens_per_sec),
            "num_tokens": data['runs'][0]['num_tokens'],
            "num_chars": data['runs'][0]['num_chars']
        }
        
        print(f"{data['name']}:")
        print(f"  Load time:     {stats[variant_key]['load_time']['mean']:>8.2f} ± {stats[variant_key]['load_time']['stdev']:>6.2f} ms")
        print(f"  Encode time:   {stats[variant_key]['encode_time']['mean']:>8.2f} ± {stats[variant_key]['encode_time']['stdev']:>6.2f} ms")
        print(f"  Tokens/sec:    {stats[variant_key]['tokens_per_sec']['mean']:>8.0f} ± {stats[variant_key]['tokens_per_sec']['stdev']:>6.0f}")
        print(f"  Tokens:        {stats[variant_key]['num_tokens']}")
        print()
    
    # Generate TSV report
    output_file = BENCHMARKS_DIR / "benchmark_results.tsv"
    print(f"=== Generating TSV report: {output_file} ===")
    
    with open(output_file, 'w') as f:
        # Header
        f.write("Variant\tLoad Time (ms)\tLoad Time StdDev\tEncode Time (ms)\tEncode Time StdDev\t")
        f.write("Tokens/sec\tTokens/sec StdDev\tNum Tokens\tNum Chars\n")
        
        # Data rows
        for variant_key in VARIANTS.keys():
            if variant_key not in stats:
                continue
            
            s = stats[variant_key]
            f.write(f"{s['name']}\t")
            f.write(f"{s['load_time']['mean']:.2f}\t{s['load_time']['stdev']:.2f}\t")
            f.write(f"{s['encode_time']['mean']:.2f}\t{s['encode_time']['stdev']:.2f}\t")
            f.write(f"{s['tokens_per_sec']['mean']:.0f}\t{s['tokens_per_sec']['stdev']:.0f}\t")
            f.write(f"{s['num_tokens']}\t{s['num_chars']}\n")
    
    print(f"✓ Report saved to {output_file}")
    print()
    
    # Also save raw JSON data
    json_file = BENCHMARKS_DIR / "benchmark_results.json"
    with open(json_file, 'w') as f:
        json.dump({
            "config": {
                "num_runs": NUM_RUNS,
                "input_file": str(INPUT_FILE),
                "tokenizer_file": str(TOKENIZER_FILE)
            },
            "results": all_results,
            "statistics": stats
        }, f, indent=2)
    
    print(f"✓ Raw data saved to {json_file}")
    print()
    print("=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
