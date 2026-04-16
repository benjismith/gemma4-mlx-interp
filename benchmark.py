"""Basic latency and throughput benchmarks for Gemma 4 E4B on this hardware.

Measures:
  1. mlx_vlm.generate: prefill tok/s, generation tok/s, TTFT
  2. forward.py: single forward-pass latency (no generation loop)
  3. hooks.py: instrumented forward-pass latency (the cost we pay per experiment)

Run from project root:
    python benchmark.py
"""

import time

import mlx.core as mx

from forward import forward as plain_forward
from forward import load_model, _tokenize
from hooks import run_with_cache
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template


PROMPTS = [
    ("short", "What is the capital of France?"),
    ("medium", "Explain how a transformer model processes a sentence, step by step."),
    ("long", "Write a detailed comparison of Python and Rust for systems programming, "
     "covering performance, safety, ecosystem, learning curve, and use cases. "
     "Give concrete examples for each point."),
]

MAX_TOKENS = 128


def bench_generate(model, processor):
    """Benchmark mlx_vlm.generate for prefill + generation throughput."""
    print("=" * 70)
    print("mlx_vlm.generate benchmarks")
    print("=" * 70)

    for label, prompt in PROMPTS:
        result = generate(
            model, processor, prompt,
            max_tokens=MAX_TOKENS, temperature=0.0, verbose=False,
        )
        print(f"\n  [{label}] {prompt[:60]}...")
        print(f"    Prompt tokens:  {result.prompt_tokens}")
        print(f"    Gen tokens:     {result.generation_tokens}")
        print(f"    Prefill:        {result.prompt_tps:>8.1f} tok/s")
        print(f"    Generation:     {result.generation_tps:>8.1f} tok/s")
        print(f"    TTFT:           {result.prompt_tokens / result.prompt_tps * 1000:>8.1f} ms")
        print(f"    Peak memory:    {result.peak_memory:>8.2f} GB")


def bench_forward(model, processor):
    """Benchmark plain forward pass (no generation loop)."""
    print("\n" + "=" * 70)
    print("forward.py single-pass benchmarks")
    print("=" * 70)

    for label, prompt in PROMPTS:
        input_ids = _tokenize(processor, model, prompt)
        n_tokens = input_ids.shape[1]

        # Warmup
        logits, _ = plain_forward(model, processor, prompt)
        mx.eval(logits)

        # Timed run
        t0 = time.perf_counter()
        logits, _ = plain_forward(model, processor, prompt)
        mx.eval(logits)
        elapsed = time.perf_counter() - t0

        print(f"\n  [{label}] {n_tokens} input tokens")
        print(f"    Latency:        {elapsed * 1000:>8.1f} ms")
        print(f"    Throughput:     {n_tokens / elapsed:>8.1f} tok/s (input)")


def bench_hooks(model, processor):
    """Benchmark instrumented forward pass with activation caching."""
    print("\n" + "=" * 70)
    print("hooks.py run_with_cache benchmarks")
    print("=" * 70)

    for label, prompt in PROMPTS:
        input_ids = _tokenize(processor, model, prompt)
        n_tokens = input_ids.shape[1]

        # Warmup
        logits, cache = run_with_cache(model, input_ids)

        # Timed run
        t0 = time.perf_counter()
        logits, cache = run_with_cache(model, input_ids)
        elapsed = time.perf_counter() - t0

        cache_bytes = sum(v.nbytes for v in cache.values())
        print(f"\n  [{label}] {n_tokens} input tokens")
        print(f"    Latency:        {elapsed * 1000:>8.1f} ms")
        print(f"    Throughput:     {n_tokens / elapsed:>8.1f} tok/s (input)")
        print(f"    Cache entries:  {len(cache)}")
        print(f"    Cache size:     {cache_bytes / 1024 / 1024:>8.1f} MB")
        print(f"    Overhead vs plain forward: measured in next section")

    # Direct overhead comparison on the medium prompt
    print("\n  --- Overhead comparison (medium prompt, 5 runs each) ---")
    _, medium_prompt = PROMPTS[1]
    input_ids = _tokenize(processor, model, medium_prompt)

    plain_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        logits, _ = plain_forward(model, processor, medium_prompt)
        mx.eval(logits)
        plain_times.append(time.perf_counter() - t0)

    hooked_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        logits, cache = run_with_cache(model, input_ids)
        hooked_times.append(time.perf_counter() - t0)

    avg_plain = sum(plain_times) / len(plain_times)
    avg_hooked = sum(hooked_times) / len(hooked_times)
    overhead = (avg_hooked - avg_plain) / avg_plain * 100

    print(f"    Plain avg:      {avg_plain * 1000:>8.1f} ms")
    print(f"    Hooked avg:     {avg_hooked * 1000:>8.1f} ms")
    print(f"    Overhead:       {overhead:>+7.1f}%")


if __name__ == "__main__":
    print("Loading model...")
    model, processor = load_model()

    bench_generate(model, processor)
    bench_forward(model, processor)
    bench_hooks(model, processor)

    print("\n" + "=" * 70)
    print("Done.")
