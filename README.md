# Local SLM Benchmark

A benchmarking suite for small language models running locally with Ollama. Compare inference performance, memory usage, and output quality across multiple models.

## Features

- **Local Inference**: Run models entirely offline via Ollama
- **Structured Output**: JSON schema enforcement with Pydantic validation + retry
- **Model Comparison**: Benchmark 3+ models on identical hardware
- **Temperature Analysis**: Document variance at different temperature settings

## Models Tested

- Llama 3.2 3B
- Phi-4 Mini
- Mistral 7B

## Benchmark Results (Apple M-series, macOS)

| Model | Tokens/sec | Time to First Token | Avg Response Time | JSON Valid |
|-------|-----------|-------------------|------------------|-----------|
| llama3.2:3b | 48.9 | 0.56s | 7.07s | 100% |
| phi4-mini | 46.1 | 0.84s | 6.72s | 100% |
| mistral:7b | 23.7 | 1.23s | 11.56s | 100% |

Key findings:
- 3B-class models deliver ~2x throughput vs 7B on the same hardware
- llama3.2:3b is fastest on TPS and TTFT — best for latency-sensitive use cases
- phi4-mini produces more concise answers, resulting in shortest total response time
- All three models achieve 100% JSON schema validity with Pydantic enforcement

## Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b

# Install Python dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Run single model inference with performance metrics
python src/inference.py --model llama3.2:3b --prompt "Explain quantum computing"

# Run structured output with Pydantic validation + retry
python src/structured_output.py --model llama3.2:3b

# Compare temperature variance (same prompt, different randomness)
python src/inference.py --model llama3.2:3b --prompt "Explain TCP vs UDP" --temperature 0
python src/inference.py --model llama3.2:3b --prompt "Explain TCP vs UDP" --temperature 0.7

# Run full benchmark suite (all 3 models, all tests)
python src/benchmark.py
# Results saved to reports/benchmark_results.json

# Run tests
pytest tests/ -v
```

## Benchmarks Measured

- Tokens per second (TPS)
- Time to first token (TTFT)
- Total response latency
- Memory usage (RSS)
- JSON validity rate
- Temperature variance analysis
