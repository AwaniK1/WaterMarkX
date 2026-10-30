"""Shared test setup: never load the real similarity model in unit tests."""

import os

import pytest

import similarity


@pytest.fixture(autouse=True)
def fake_similarity(monkeypatch, request):
    # The real-model integration test opts out by name.
    if os.environ.get("RUN_INTEGRATION") == "1" and "integration" in request.node.nodeid:
        return
    monkeypatch.setattr(similarity, "semantic_similarity", lambda a, b: 0.9)
