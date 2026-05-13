"""Structured output with JSON schema enforcement, Pydantic validation, and retry."""

import argparse
import json

import ollama
from pydantic import BaseModel, ValidationError


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    description: str
    confidence: float


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    summary: str


EXTRACTION_PROMPT = """Extract all named entities from the following text.
Return a JSON object with this exact schema:
{{
  "entities": [
    {{
      "name": "entity name",
      "entity_type": "PERSON | ORG | LOCATION | DATE | OTHER",
      "description": "brief description",
      "confidence": 0.0 to 1.0
    }}
  ],
  "summary": "one sentence summary of the text"
}}

Text: {text}

Return ONLY valid JSON, no other text."""


def extract_with_retry(model: str, text: str, max_retries: int = 1) -> ExtractionResult | None:
    """Extract structured data with validation and retry on failure."""
    prompt = EXTRACTION_PROMPT.format(text=text)

    for attempt in range(max_retries + 1):
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
            format="json",
        )

        raw = response["message"]["content"]
        try:
            parsed = json.loads(raw)
            result = ExtractionResult(**parsed)
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                prompt = (
                    f"Your previous response was invalid JSON or didn't match the schema. "
                    f"Error: {e}\n\nPlease try again.\n\n{prompt}"
                )
            continue

    print("All attempts failed. Returning None.")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--text", default="Apple CEO Tim Cook announced new AI features at WWDC in Cupertino on June 10, 2025.")
    args = parser.parse_args()

    result = extract_with_retry(args.model, args.text)
    if result:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print("Extraction failed after retries.")


if __name__ == "__main__":
    main()
