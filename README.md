# Deep Research Agent

A full-stack AI research agent app for deep research using multiple LLMs, with a modern chat UI and real-time streaming results.

## Features
- **Multi-LLM Research:** Select and query multiple large language models (ChatGPT, Gemini, Grok, etc.) in parallel.
- **Leading Model Summarization:** Choose a leading model to synthesize and conclude based on all model outputs.
- **Streaming Results:** See model responses and summary/conclusion streamed live as they are ready.
- **Rich Chat UI:** React + Vite frontend with Markdown rendering, copy-to-clipboard, and responsive design.
- **Async Python Backend:** FastAPI backend with async API calls, error handling, and CORS support.
- **Secure API Key Management:** API keys are stored in a separate, git-ignored file for security.

## Project Structure
```
research-agent/
├── backend/              # FastAPI backend
│   ├── main.py           # Main API logic
│   ├── api_keys.py       # API keys (excluded from git)
│   ├── requirements.txt  # Python dependencies
│   └── README.md         # Backend usage
├── frontend/             # React + Vite + TypeScript frontend
│   ├── src/              # Frontend source code
│   ├── index.html        # App entry (uses research.png as favicon)
│   ├── research.png      # App icon
│   └── README.md         # Frontend usage
├── Makefile              # Start/stop/restart both backend and frontend
├── .gitignore            # Excludes venv, node_modules, api_keys.py, etc.
└── README.md             # (This file)
```

## Quick Start

### 1. Backend (Python/FastAPI)
```bash
cd backend
python3 -m venv ../backend-venv
source ../backend-venv/bin/activate
pip install -r requirements.txt
# Copy api_keys.py.example to api_keys.py and add your API keys
uvicorn main:app --reload
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
- All API keys are stored in `backend/api_keys.py` (excluded from git).
- You must provide your own keys for OpenAI, Gemini, Grok, etc.

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
- **Never commit your API keys** (`api_keys.py` is git-ignored).
- The Python virtual environment (`backend-venv/`) and frontend `node_modules/` are also git-ignored.
- Use best practices for async API calls, error handling, and user experience (see `.github/copilot-instructions.md`).

## License
This project is for research and educational purposes. Please review the licenses of any LLM APIs you use.

---

For more details, see the `README.md` files in the `backend/` and `frontend/` folders.
