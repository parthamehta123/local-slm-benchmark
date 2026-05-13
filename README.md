# Local SLM Benchmark

A benchmarking suite for small language models running locally with Ollama. Compare inference performance, memory usage, and output quality across multiple models.

## Features

- **Local Inference**: Run models entirely offline via Ollama
- **Structured Output**: JSON schema enforcement with Pydantic validation + retry
- **Model Comparison**: Benchmark 3+ models on identical hardware
- **Temperature Analysis**: Document variance at different temperature settings
- **Quantization Study**: Compare GGUF Q4/Q5 quality vs speed tradeoffs

## Models Tested

- Llama 3.2 3B
- Phi-4 Mini
- Mistral 7B
- Quantized variants (Q4, Q5, Q8)

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
```

## Usage

```bash
# Run single model inference
python src/inference.py --model llama3.2:3b --prompt "Explain quantum computing"

# Run structured output with validation
python src/structured_output.py --model llama3.2:3b

# Run full benchmark suite
python src/benchmark.py

# Generate comparison report
python src/report.py
```

## Benchmarks Measured

- Tokens per second
- Time to first token (TTFT)
- Total response latency
- Memory usage (RSS)
- JSON validity rate
- Output quality scoring on standardized prompts
