"""Tests for inference measurement."""

from unittest.mock import MagicMock, patch

from src.inference import measure_inference


@patch("src.inference.ollama")
@patch("src.inference.psutil")
def test_measure_inference_returns_all_fields(mock_psutil, mock_ollama):
    mock_process = MagicMock()
    mock_process.memory_info.return_value.rss = 500 * 1024 * 1024
    mock_psutil.Process.return_value = mock_process

    mock_ollama.chat.return_value = iter(
        [
            {"message": {"content": "Hello "}},
            {"message": {"content": "world!"}},
        ]
    )

    result = measure_inference("test-model", "Say hello")

    assert result["model"] == "test-model"
    assert "total_time_s" in result
    assert "ttft_s" in result
    assert "tokens_per_second" in result
    assert "memory_before_mb" in result
    assert "memory_after_mb" in result
    assert result["response"] == "Hello world!"
    assert result["temperature"] == 0.0


@patch("src.inference.ollama")
@patch("src.inference.psutil")
def test_measure_inference_with_custom_temperature(mock_psutil, mock_ollama):
    mock_process = MagicMock()
    mock_process.memory_info.return_value.rss = 500 * 1024 * 1024
    mock_psutil.Process.return_value = mock_process

    mock_ollama.chat.return_value = iter([{"message": {"content": "Hi"}}])

    result = measure_inference("test-model", "Say hi", temperature=0.7)
    assert result["temperature"] == 0.7
