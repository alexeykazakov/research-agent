# Makefile for Deep Research Agent

.PHONY: start stop restart backend frontend check-backend check-python clean setup check-npm

check-python:
	@echo "Checking Python version..."
	@if ! command -v python3.12 > /dev/null; then \
		echo "Python 3.12 is required but not found. Please install it first:"; \
		echo "sudo apt update && sudo apt install python3.12 python3.12-venv"; \
		exit 1; \
	fi

check-npm:
	@echo "Checking npm dependencies..."
	@if [ ! -f "backend/package.json" ]; then \
		echo "Initializing Node.js project..." && \
		cd backend && npm init -y; \
	fi
	@if [ ! -f "backend/node_modules/@modelcontextprotocol/server-brave-search/dist/index.js" ]; then \
		echo "Installing MCP Brave Search server locally..." && \
		cd backend && npm install @modelcontextprotocol/server-brave-search; \
	fi

check-backend:
	@echo "Checking if backend is running..."
	@if ! curl -s http://localhost:8000/models > /dev/null; then \
		echo "Backend is not running or not responding"; \
		exit 1; \
	fi

clean:
	@echo "Cleaning up old virtual environments..."
	rm -rf backend/.venv
	rm -rf backend/node_modules
	rm -f backend/package.json backend/package-lock.json
	@echo "Cleanup complete."

setup: check-npm
	@echo "Setup complete."

start: check-python clean setup
	@echo "Starting backend (FastAPI) and frontend (Vite)..."
	@if ! lsof -i:8000 | grep LISTEN > /dev/null; then \
		echo "Starting backend server..." && \
		cd backend && \
		uv venv --python=python3.12 && \
		. .venv/bin/activate && \
		uv pip install -r requirements.txt && \
		uvicorn main:app --reload --host 0.0.0.0 --port 8000 & \
		sleep 5 && \
		$(MAKE) check-backend || (echo "Backend failed to start" && exit 1); \
	fi
	@if ! lsof -i:5173 | grep LISTEN > /dev/null; then \
		echo "Starting frontend server..." && \
		cd frontend && \
		npm install && \
		npm run dev & \
		sleep 5 && \
		if ! curl -s http://localhost:5173 > /dev/null; then \
			echo "Frontend failed to start" && exit 1; \
		fi; \
	fi
	@echo "Both backend and frontend are running."

stop:
	@echo "Stopping backend (FastAPI) and frontend (Vite)..."
	-@pkill -f "uvicorn main:app"
	-@pkill -f "npm run dev"
	@echo "Both backend and frontend should be stopped."

restart: clean
	$(MAKE) stop
	$(MAKE) start

backend: check-python clean setup
	@echo "Starting backend server..."
	cd backend && \
	uv venv --python=python3.12 && \
	. .venv/bin/activate && \
	uv pip install -r requirements.txt && \
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

frontend:
	@echo "Starting frontend server..."
	cd frontend && \
	npm install && \
	npm run dev
