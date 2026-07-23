"""Tests for the main module."""

from src.main import greet


def test_greet() -> None:
    """Test the greet function."""
    assert greet("42") == "Hello, 42!"
