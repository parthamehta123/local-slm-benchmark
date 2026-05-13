"""Tests for structured output validation and retry."""

import json
from unittest.mock import patch

from src.structured_output import (
    ExtractionResult,
    ExtractedEntity,
    extract_with_retry,
)


def test_extraction_result_valid():
    result = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Google",
                entity_type="ORG",
                description="Tech company",
                confidence=0.95,
            )
        ],
        summary="A sentence about Google.",
    )
    assert len(result.entities) == 1
    assert result.entities[0].name == "Google"


def test_extraction_result_empty_entities():
    result = ExtractionResult(entities=[], summary="No entities found.")
    assert result.entities == []


@patch("src.structured_output.ollama")
def test_extract_with_retry_valid_response(mock_ollama):
    valid_json = json.dumps(
        {
            "entities": [
                {
                    "name": "Apple",
                    "entity_type": "ORG",
                    "description": "Technology company",
                    "confidence": 0.9,
                }
            ],
            "summary": "About Apple.",
        }
    )
    mock_ollama.chat.return_value = {"message": {"content": valid_json}}

    result = extract_with_retry("test-model", "Apple released a new product.")
    assert result is not None
    assert result.entities[0].name == "Apple"


@patch("src.structured_output.ollama")
def test_extract_with_retry_invalid_then_valid(mock_ollama):
    valid_json = json.dumps(
        {
            "entities": [],
            "summary": "Test.",
        }
    )
    mock_ollama.chat.side_effect = [
        {"message": {"content": "not json at all"}},
        {"message": {"content": valid_json}},
    ]

    result = extract_with_retry("test-model", "Some text", max_retries=1)
    assert result is not None


@patch("src.structured_output.ollama")
def test_extract_with_retry_all_fail(mock_ollama):
    mock_ollama.chat.return_value = {"message": {"content": "bad json"}}

    result = extract_with_retry("test-model", "Some text", max_retries=1)
    assert result is None
