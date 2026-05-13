"""Full benchmark suite: compare multiple models on standardized prompts."""

import json
import time
from pathlib import Path

from tabulate import tabulate

from src.inference import measure_inference
from src.structured_output import extract_with_retry

MODELS = [
    "llama3.2:3b",
    "phi4-mini",
    "mistral:7b",
]

TEST_PROMPTS = [
    "Explain the difference between TCP and UDP in 3 sentences.",
    "What are the SOLID principles in software engineering?",
    "Write a Python function to check if a string is a palindrome.",
    "Summarize the concept of gradient descent in machine learning.",
    "What is the CAP theorem in distributed systems?",
]

EXTRACTION_TEXTS = [
    "Google CEO Sundar Pichai announced Gemini 2.0 at their Mountain View headquarters on December 11, 2024.",
    "The Federal Reserve raised interest rates by 25 basis points on March 19, 2025, according to Chair Jerome Powell.",
]


def run_benchmark():
    """Run complete benchmark across all models and prompts."""
    results = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model}")
        print(f"{'='*60}")

        model_metrics = {
            "model": model,
            "avg_tokens_per_second": 0,
            "avg_ttft_s": 0,
            "avg_total_time_s": 0,
            "json_validity_rate": 0,
        }

        # Inference benchmarks
        inference_results = []
        for prompt in TEST_PROMPTS:
            print(f"  Prompt: {prompt[:50]}...")
            result = measure_inference(model, prompt, temperature=0)
            inference_results.append(result)
            time.sleep(0.5)

        model_metrics["avg_tokens_per_second"] = round(
            sum(r["tokens_per_second"] for r in inference_results) / len(inference_results), 1
        )
        model_metrics["avg_ttft_s"] = round(
            sum(r["ttft_s"] for r in inference_results) / len(inference_results), 3
        )
        model_metrics["avg_total_time_s"] = round(
            sum(r["total_time_s"] for r in inference_results) / len(inference_results), 3
        )

        # Structured output benchmarks
        valid = 0
        for text in EXTRACTION_TEXTS:
            result = extract_with_retry(model, text, max_retries=1)
            if result is not None:
                valid += 1
        model_metrics["json_validity_rate"] = round(valid / len(EXTRACTION_TEXTS), 2)

        # Temperature variance test
        temp_results = {}
        test_prompt = TEST_PROMPTS[0]
        for temp in [0.0, 0.7]:
            responses = []
            for _ in range(3):
                r = measure_inference(model, test_prompt, temperature=temp)
                responses.append(r["response"])
            temp_results[f"temp_{temp}"] = responses

        model_metrics["temperature_variance"] = temp_results
        results.append(model_metrics)

    # Summary table
    table_data = []
    for r in results:
        table_data.append([
            r["model"],
            r["avg_tokens_per_second"],
            r["avg_ttft_s"],
            r["avg_total_time_s"],
            r["json_validity_rate"],
        ])

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(tabulate(
        table_data,
        headers=["Model", "Avg TPS", "Avg TTFT(s)", "Avg Total(s)", "JSON Valid %"],
        tablefmt="grid",
    ))

    # Save results
    output_path = Path("reports/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
