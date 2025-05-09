# Deep Research Agent Backend

This is the FastAPI backend for the Deep Research Agent app.

## Features
- Provides a list of available LLM models
- Accepts research prompts and distributes them to selected models
- Aggregates results and uses a leading model to summarize and conclude
- Async API calls for fast response
- CORS enabled for frontend integration

## Getting Started

1. Create and activate the Python virtual environment (if not already):
   ```bash
   uv venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Run the backend server:
   ```bash
   uvicorn backend.main:app --reload
   ```
4. The API will be available at http://localhost:8000

## Endpoints
- `GET /models` — List available models
- `POST /prompt` — Submit a prompt and get aggregated results

---

For frontend setup, see the frontend/README.md.
