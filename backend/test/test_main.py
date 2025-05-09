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

@pytest.mark.asyncio
async def test_prompt_multiple_models(monkeypatch):
    from backend import main
    async def dummy_call_model_api(model, prompt, think, research):
        return f"Dummy response for {model}"
    monkeypatch.setattr(main, "call_model_api", dummy_call_model_api)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/prompt", json={
            "prompt": "What is AI?",
            "models": ["ChatGPT 4", "DeepSeek (Ollama local)"],
            "leading_model": "ChatGPT 4",
            "think": True,
            "research": False
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"][0]["response"].startswith("Dummy response")
        assert "summary" in data
        assert "conclusion" in data

@pytest.mark.asyncio
async def test_prompt_stream(monkeypatch):
    from backend import main
    async def dummy_call_model_api(model, prompt, think, research):
        return f"Dummy response for {model}"
    monkeypatch.setattr(main, "call_model_api", dummy_call_model_api)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        params = {
            "prompt": "Stream test?",
            "models": ["ChatGPT 4"],
            "leading_model": "ChatGPT 4",
            "think": True,
            "web_search": False
        }
        resp = await ac.get("/prompt/stream", params=params)
        assert resp.status_code == 200
        # Should be a streaming response (text/event-stream)
        assert resp.headers["content-type"].startswith("text/event-stream")
        # Read a few lines from the stream
        lines = [line for line in resp.text.splitlines() if line.strip()]
        assert any("Dummy response" in l for l in lines)

import pytest
import asyncio
from backend.main import RateLimiter

@pytest.mark.asyncio
async def test_rate_limiter_wait(monkeypatch):
    rl = RateLimiter(requests_per_second=1000)  # very high rate, should not wait
    called = []
    orig_sleep = asyncio.sleep
    async def fake_sleep(secs):
        called.append(secs)
        await orig_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await rl.acquire("test-model")
    await rl.acquire("test-model")
    # Should not have called sleep (or only with 0)
    assert all(s == 0 for s in called)

@pytest.mark.asyncio
async def test_rate_limiter_enforces_wait(monkeypatch):
    rl = RateLimiter(requests_per_second=1)  # 1 per second, should wait
    called = []
    orig_sleep = asyncio.sleep
    async def fake_sleep(secs):
        called.append(secs)
        await orig_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await rl.acquire("test-model")
    await rl.acquire("test-model")
    # Should have called sleep with a value > 0
    assert any(s > 0 for s in called)