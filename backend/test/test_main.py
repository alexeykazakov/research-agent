# ...existing code...
import pytest
from httpx import AsyncClient
from backend.main import app

import sys
import os
import asyncio

import pytest

@pytest.mark.asyncio
async def test_get_models():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.get("/models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert "ChatGPT 4" in resp.json()

@pytest.mark.asyncio
async def test_prompt_validation():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # No models selected
        resp = await ac.post("/prompt", json={
            "prompt": "Test prompt",
            "models": [],
            "leading_model": "",
            "think": True,
            "research": False
        })
        assert resp.status_code == 400

@pytest.mark.asyncio
async def test_prompt_dummy_model(monkeypatch):
    # Patch call_model_api to avoid real API calls
    from backend import main
    async def dummy_call_model_api(model, prompt, think, research):
        return f"Dummy response for {model}"
    monkeypatch.setattr(main, "call_model_api", dummy_call_model_api)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/prompt", json={
            "prompt": "Why is the sky blue?",
            "models": ["DeepSeek (Ollama local)"],
            "leading_model": "DeepSeek (Ollama local)",
            "think": True,
            "research": False
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert data["results"][0]["response"].startswith("Dummy response")