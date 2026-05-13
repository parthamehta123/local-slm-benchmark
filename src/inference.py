"""Basic inference wrapper around Ollama with performance measurement."""

import argparse
import time

import ollama
import psutil


def measure_inference(model: str, prompt: str, temperature: float = 0.0) -> dict:
    """Run inference and measure performance metrics."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    start = time.perf_counter()
    first_token_time = None
    tokens = []

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={"temperature": temperature},
    )

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        token = chunk["message"]["content"]
        tokens.append(token)

    end = time.perf_counter()
    mem_after = process.memory_info().rss / 1024 / 1024

    full_response = "".join(tokens)
    total_time = end - start
    ttft = first_token_time - start if first_token_time else total_time

    # Approximate token count (rough: ~4 chars per token)
    approx_tokens = len(full_response) // 4

    return {
        "model": model,
        "prompt": prompt[:100],
        "response": full_response,
        "total_time_s": round(total_time, 3),
        "ttft_s": round(ttft, 3),
        "approx_tokens": approx_tokens,
        "tokens_per_second": round(approx_tokens / total_time, 1)
        if total_time > 0
        else 0,
        "memory_before_mb": round(mem_before, 1),
        "memory_after_mb": round(mem_after, 1),
        "temperature": temperature,
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference with Ollama")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    result = measure_inference(args.model, args.prompt, args.temperature)
    for key, value in result.items():
        if key != "response":
            print(f"{key}: {value}")
    print(f"\nResponse:\n{result['response']}")


if __name__ == "__main__":
    main()
