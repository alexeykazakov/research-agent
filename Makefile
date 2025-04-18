# Makefile for Deep Research Agent

.PHONY: start stop restart backend frontend

start:
	@echo "Starting backend (FastAPI) and frontend (Vite)..."
	@if ! lsof -i:8000 | grep LISTEN > /dev/null; then \
		. backend-venv/bin/activate && uvicorn backend.main:app --reload & \
	fi
	@if ! lsof -i:5173 | grep LISTEN > /dev/null; then \
		cd frontend && npm run dev & \
	fi
	@echo "Both backend and frontend should be running."

stop:
	@echo "Stopping backend (FastAPI) and frontend (Vite)..."
	-@pkill -f "uvicorn backend.main:app"
	-@pkill -f "npm run dev"
	@echo "Both backend and frontend should be stopped."

restart:
	$(MAKE) stop
	$(MAKE) start

backend:
	. backend-venv/bin/activate && uvicorn backend.main:app --reload

frontend:
	cd frontend && npm run dev
