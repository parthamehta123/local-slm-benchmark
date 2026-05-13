FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest pytest-cov

COPY . .

ENV OLLAMA_HOST=http://ollama:11434

CMD ["python", "-m", "pytest", "tests/", "-v"]
