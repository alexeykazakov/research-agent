from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import httpx
from fastapi.responses import StreamingResponse
import json
from typing import AsyncGenerator
import os
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import time

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy model list
MODEL_CONFIG = {
    "ChatGPT 4": {
        "api_id": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY")
    },
    "ChatGPT 4.1": {
        "api_id": "gpt-4-1106-preview",
        "api_key": os.environ.get("OPENAI_API_KEY")
    },
    "ChatGPT 4o": {
        "api_id": "gpt-4o",
        "api_key": os.environ.get("OPENAI_API_KEY")
    },
    "Grok 3": {
        "api_id": "grok-3-latest",
        "api_key": os.environ.get("GROK_API_KEY")
    },
    "Gemini 2.0 Flash": {
        "api_id": "gemini-2.0-flash",
        "api_key": os.environ.get("GEMINI_API_KEY")
    },
    "Gemini 2.5 Pro": {
        "api_id": "gemini-2.5-pro-exp-03-25",
        "api_key": os.environ.get("GEMINI_API_KEY")
    },
    "Gemini 2.5 Flash": {
        "api_id": "models/gemini-2.5-flash-preview-04-17",
        "api_key": os.environ.get("GEMINI_API_KEY")
    },
    "DeepSeek (Ollama local)": {
        "api_id": "deepseek-coder",
        "api_key": "ollama-local-key"
    },
    "DeepSeek API": {
        "api_id": "deepseek-api",
        "api_key": "sk-xxx-deepseekapi"
    }
}

# API keys
MODEL_KEYS = {model: config["api_key"] for model, config in MODEL_CONFIG.items()}

class PromptRequest(BaseModel):
    prompt: str
    models: List[str]
    leading_model: str
    think: bool
    web_search: bool = True

class ModelResponse(BaseModel):
    model: str
    response: str

class AggregatedResponse(BaseModel):
    results: List[ModelResponse]
    summary: str
    conclusion: str

# Add rate limiter class at the top of the file, after imports
class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = asyncio.Lock()
        self.model_locks = {}  # Track locks per model

    async def acquire(self, model_name: str) -> None:
        # Get or create a lock for this specific model
        if model_name not in self.model_locks:
            self.model_locks[model_name] = asyncio.Lock()
        
        async with self.model_locks[model_name]:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.info(f"Rate limit: Waiting {wait_time:.2f}s for model {model_name}")
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()
            logger.info(f"Rate limit acquired for model {model_name}")

# Create a single rate limiter instance for all models
rate_limiter = RateLimiter(requests_per_second=20.0)  # 20 requests per second

@app.get("/models", response_model=List[str])
def get_models():
    return list(MODEL_CONFIG.keys())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper for MCPAgent logic ---
async def run_mcp_agent(model_name, llm, client, prompt, rate_limiter, max_steps=30):
    try:
        logger.info(f"Waiting for rate limit for model {model_name}")
        await rate_limiter.acquire(model_name)
        logger.info(f"Rate limit acquired for model {model_name}, starting research")
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=max_steps
        )
        logger.info("Running MCP agent with prompt...")
        result = await agent.run(prompt)
        logger.info("Research completed successfully")
        return str(result)
    except Exception as e:
        logger.error("Error during research: %s", str(e), exc_info=True)
        if "RATE_LIMITED" in str(e):
            return "I apologize, but I'm currently unable to access real-time information due to rate limiting. Please try again in a few moments."
        return f"[Research Error] {str(e)}"

# --- Helper for MCPClient creation ---
def make_mcp_client(config):
    logger.info("Initializing MCP client with config: %s", config)
    return MCPClient.from_dict(config)

async def call_model_api(model: str, prompt: str, think: bool, web_search: bool) -> str:
    """
    Call the real API for the given model. Uses OpenAI for ChatGPT models.
    """
    if web_search:
        logger.info(f"Web search mode enabled for model {model}")
        config = {
            # Use MCPAgent with Brave Search MCP server for research-enabled models
            "mcpServers": {
                "brave": {
                    "command": "node",
                    "args": ["node_modules/@modelcontextprotocol/server-brave-search/dist/index.js"],
                    "env": {
                        "BRAVE_API_KEY": os.getenv("BRAVE_SEARCH_API_KEY")
                    },
                    "cwd": os.path.dirname(__file__)
                }
            }
        }

        # Create system message that explicitly requires using search tools
        system_message = "You are an expert research assistant with access to real-time information through search tools.\n"
        if not think:
            system_message += "Do not do any reasoning/thinking, just answer directly.\n"
        system_message += (
            "IMPORTANT: You MUST use the search tools available to you to find current and accurate information.\n"
            "DO NOT rely on your training data for time-sensitive information.\n"
            "For ANY question about:\n"
            "- Current dates and times\n"
            "- Recent events and news\n"
            "- Current prices and market data\n"
            "- Any information that might change over time\n"
            "You MUST:\n"
            "1. First use the search tools to find the current information\n"
            "2. Then provide the answer based on the search results\n"
            "3. Never assume you know the current date or time without searching\n"
            "If you're asked about today's date, you MUST search for it first.\n"
            "If you're asked about current events, you MUST search for them first.\n"
            "If you're asked about any time-sensitive information, you MUST search for it first.\n"
            "Remember: Your training data is not up-to-date. Always use search tools for current information."
        )

        # Create a prompt that includes the system message and explicit search instruction
        full_prompt = f"""
{system_message}
User question: {prompt}

IMPORTANT INSTRUCTIONS:
1. You MUST use the search tool to find out about the current events
2. DO NOT rely on your training data for current events
3. DO NOT make assumptions about the current events
4. You MUST search first, then answer
5. If you cannot search, say "I need to search for the current events but am unable to do so"

Remember: Your training data is outdated. You MUST search for the current events.
"""

        if model in {"ChatGPT 4", "ChatGPT 4.1", "ChatGPT 4o"}:
            # Use MODEL_CONFIG to get the api_id for ChatGPT models
            llm = ChatOpenAI(
                model=MODEL_CONFIG[model]["api_id"],
                temperature=0,
            )
            client = make_mcp_client(config)
            return await run_mcp_agent(model, llm, client, full_prompt, rate_limiter, max_steps=30)
        elif model == "Grok 3":
            llm = ChatXAI(
                model="grok-3-latest",
                api_key=MODEL_CONFIG[model]["api_key"],
                temperature=0,
            )
            client = make_mcp_client(config)
            return await run_mcp_agent(model, llm, client, full_prompt, rate_limiter, max_steps=30)
        elif model in {"Gemini 2.0 Flash", "Gemini 2.5 Pro", "Gemini 2.5 Flash"}:
            # Use MODEL_CONFIG to get the api_id for Gemini models
            llm = ChatGoogleGenerativeAI(
                model=MODEL_CONFIG[model]["api_id"],
                temperature=0,
                google_api_key=MODEL_CONFIG[model]["api_key"]
            )
            client = make_mcp_client(config)
            return await run_mcp_agent(model, llm, client, full_prompt, rate_limiter, max_steps=30)
    
    # Original model-specific implementations for non-research mode
    if model in {"ChatGPT 4", "ChatGPT 4.1", "ChatGPT 4o"}:
        # OpenAI Chat API
        import httpx
        system_prompt = """You are an expert research assistant.{}{}""".format(
            " Do not do any reasoning/thinking, just answer directly." if not think else "",
            " Use all available tools to do deep research." if web_search else ""
        )
        headers = {
            "Authorization": f"Bearer {MODEL_CONFIG[model]['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODEL_CONFIG[model]["api_id"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    elif model in {"Gemini 2.0 Flash", "Gemini 2.5 Pro"}:
        # Use google-genai SDK for Gemini models
        import asyncio
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("google-genai is not installed. Please run: pip install google-generativeai")
        
        system_prompt = "You are an expert research assistant.{}{}".format(
            " Do not do any reasoning/thinking, just answer directly." if not think else "",
            " Use all available tools to do deep research." if web_search else ""
        )
        # Configure the client
        genai.configure(api_key=MODEL_CONFIG[model]["api_key"])
        # Compose the full prompt
        full_prompt = system_prompt + "\n" + prompt
        model = genai.GenerativeModel(MODEL_CONFIG[model]["api_id"])
        # Run in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        def run_genai():
            try:
                response = model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                return f"[Gemini error] {e}"
        return await loop.run_in_executor(None, run_genai)
    elif model == "Grok 3":
        # xAI Grok 3 API integration (no tool use)
        import httpx
        system_prompt = "You are an expert research assistant.{}{}".format(
            " Do not do any reasoning/thinking, just answer directly." if not think else "",
            " Use all available tools to do deep research." if web_search else ""
        )
        headers = {
            "Authorization": f"Bearer {MODEL_CONFIG[model]['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "model": MODEL_CONFIG[model]["api_id"],
            "stream": False,
            "temperature": 1.0
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    
    # Fallback for other models (DeepSeek, etc.)
    await asyncio.sleep(1)
    return f"[{model} | Think: {think} | Research: {web_search}] {prompt} (dummy)"

@app.get("/prompt/stream")
async def prompt_stream(
    prompt: str = Query(...),
    models: List[str] = Query(...),
    leading_model: str = Query(...),
    think: bool = Query(...),
    web_search: bool = Query(True)
):
    req = PromptRequest(
        prompt=prompt,
        models=models,
        leading_model=leading_model,
        think=think,
        web_search=web_search
    )
    return StreamingResponse(stream_model_results(req), media_type="text/event-stream")

async def stream_model_results(req: PromptRequest) -> AsyncGenerator[str, None]:
    async def model_task(model):
        try:
            result = await call_model_api(model, req.prompt, req.think, req.web_search)
            return {"model": model, "response": result}
        except Exception as e:
            return {"model": model, "response": f"[Error] {str(e)}"}
    tasks = [model_task(m) for m in req.models]
    model_results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        model_results.append(result)
        yield f"data: {json.dumps(result)}\n\n"
    # After all models, get summary and conclusion from leading model
    # Pass the initial prompt and actual model results to the leading model
    summary_prompt = (
        "You are an expert research assistant. Carefully read the following research question and the answers from different AI models. "
        "Your task is to synthesize and combine the factual information and insights from all answers into a single, clear, and comprehensive summary of the research topic. "
        "Do not compare or evaluate the models themselves. Only focus on the content of their answers.\n"
        f"Research question: {req.prompt}\n"
        + "\n".join(f"{mr['model']}: {mr['response']}" for mr in model_results)
        + "\nSummary:"
    )
    summary = await call_model_api(req.leading_model, summary_prompt, True, False)
    conclusion_prompt = (
        "Based on the following research question and the combined information from all model answers, provide a clear, concise, and actionable conclusion about the research topic. "
        "Do not mention or compare the models themselves.\n"
        f"Research question: {req.prompt}\n"
        + "\n".join(f"{mr['model']}: {mr['response']}" for mr in model_results)
        + "\nConclusion:"
    )
    conclusion = await call_model_api(req.leading_model, conclusion_prompt, True, False)
    yield f"data: {json.dumps({'summary': summary, 'conclusion': conclusion})}\n\n"

@app.post("/prompt", response_model=AggregatedResponse)
async def submit_prompt(req: PromptRequest):
    if not req.models or req.leading_model not in req.models:
        raise HTTPException(status_code=400, detail="At least one model must be selected and leading model must be among them.")
    # Call all selected models in parallel
    tasks = [call_model_api(m, req.prompt, req.think, req.web_search) for m in req.models]
    results = await asyncio.gather(*tasks)
    model_responses = [ModelResponse(model=m, response=r) for m, r in zip(req.models, results)]
    # Use leading model to summarize
    summary_prompt = (
        "You are an expert research assistant. Carefully read the following answers from different AI models to the same research question. "
        "Your task is to synthesize and combine the factual information and insights from all answers into a single, clear, and comprehensive summary of the research topic. "
        "Do not compare or evaluate the models themselves. Only focus on the content of their answers.\n"
        + "\n".join(f"{mr.model}: {mr.response}" for mr in model_responses)
        + "\nSummary:"
    )
    summary_text = await call_model_api(req.leading_model, summary_prompt, True, False)
    # Use leading model to make a conclusion
    conclusion_prompt = (
        "Based on the combined information from all model answers above, provide a clear, concise, and actionable conclusion about the research topic. "
        "Do not mention or compare the models themselves.\n"
        + "\n".join(f"{mr.model}: {mr.response}" for mr in model_responses)
        + "\nConclusion:"
    )
    conclusion_text = await call_model_api(req.leading_model, conclusion_prompt, True, False)
    return AggregatedResponse(results=model_responses, summary=summary_text.strip(), conclusion=conclusion_text.strip())
