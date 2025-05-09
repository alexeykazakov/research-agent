# Deep Research Agent

A full-stack AI research agent app for deep research using multiple LLMs, with a modern chat UI and real-time streaming results.

## Features
- **Multi-LLM Research:** Select and query multiple large language models (ChatGPT, Gemini, Grok, etc.) in parallel.
- **Leading Model Summarization:** Choose a leading model to synthesize and conclude based on all model outputs.
- **Streaming Results:** See model responses and summary/conclusion streamed live as they are ready.
- **Rich Chat UI:** React + Vite frontend with Markdown rendering, copy-to-clipboard, and responsive design.
- **Async Python Backend:** FastAPI backend with async API calls, error handling, and CORS support.
- **Secure API Key Management:** API keys are stored in a separate, git-ignored file for security.
- **Web Search Integration:** Instructs LLMs to use the provided Brave web search MCP server when a web search for current events is needed.

## Project Structure
```
research-agent/
├── backend/              # FastAPI backend
│   ├── main.py           # Main API logic
│   ├── .env              # API keys (excluded from git)
│   ├── requirements.txt  # Python dependencies
│   └── README.md         # Backend usage
├── frontend/             # React + Vite + TypeScript frontend
│   ├── src/              # Frontend source code
│   ├── index.html        # App entry (uses research.png as favicon)
│   ├── research.png      # App icon
│   └── README.md         # Frontend usage
├── Makefile              # Start/stop/restart both backend and frontend
├── .gitignore            # Excludes venv, node_modules, .env, etc.
└── README.md             # (This file)
```

## Quick Start

### 1. Backend (Python/FastAPI)
```bash
cd backend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# Create .env and add your API keys
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend (React/Vite)
```bash
cd frontend
npm install
npm run dev
```

### 3. All-in-one (from project root)
```bash
make start
```

- The backend runs on [http://localhost:8000](http://localhost:8000)
- The frontend runs on [http://localhost:5173](http://localhost:5173)

## API Keys
- All API keys are stored in `backend/.env` (excluded from git).
- You must provide your own keys for OpenAI, Gemini, Grok, etc.

### Example `.env` file
```env
OPENAI_API_KEY=your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
GROK_API_KEY=your-grok-key-here
BRAVE_SEARCH_API_KEY=your-brave-search-key-here
```

## Testing

### Backend (FastAPI)
Run from the project root or backend folder:
```bash
pytest backend/test/
```

### Frontend (React)
Run from the frontend folder:
```bash
cd frontend
npm test
```

- Backend tests use pytest and httpx's AsyncClient for endpoint and logic testing.
- Frontend tests use @testing-library/react and Jest for UI/component testing.

## Security & Best Practices
- **Never commit your API keys** (`backend/.env` is git-ignored).
- The Python virtual environment (`venv/`) and frontend `node_modules/` are also git-ignored.
- Use best practices for async API calls, error handling, and user experience (see `.github/copilot-instructions.md`).

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

---

For more details, see the `README.md` files in the `backend/` and `frontend/` folders.
