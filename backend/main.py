from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import httpx
from fastapi.responses import StreamingResponse
import json
from typing import AsyncGenerator
from .api_keys import OPENAI_API_KEY, GEMINI_API_KEY, GROK_API_KEY

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
AVAILABLE_MODELS = [
    "ChatGPT 4",
    "ChatGPT 4.1",
    "ChatGPT 4o",
    "Grok 3",
    "Gemini 2.0 Flash Thinking",
    "Gemini 2.5 Pro",
    "DeepSeek (Ollama local)",
    "DeepSeek API"
]

# API keys
MODEL_KEYS = {
    "ChatGPT 4": OPENAI_API_KEY,
    "ChatGPT 4.1": OPENAI_API_KEY,
    "ChatGPT 4o": OPENAI_API_KEY,
    "Grok 3": GROK_API_KEY,
    "Gemini 2.0 Flash Thinking": GEMINI_API_KEY,
    "Gemini 2.5 Pro": GEMINI_API_KEY,
    "DeepSeek (Ollama local)": "ollama-local-key",
    "DeepSeek API": "sk-xxx-deepseekapi"
}

class PromptRequest(BaseModel):
    prompt: str
    models: List[str]
    leading_model: str
    think: bool
    research: bool

class ModelResponse(BaseModel):
    model: str
    response: str

class AggregatedResponse(BaseModel):
    results: List[ModelResponse]
    summary: str
    conclusion: str

@app.get("/models", response_model=List[str])
def get_models():
    return AVAILABLE_MODELS

async def call_model_api(model: str, prompt: str, think: bool, research: bool) -> str:
    """
    Call the real API for the given model. Uses OpenAI for ChatGPT models, Gemini API for Gemini models, and dummy for others.
    """
    if model in {"ChatGPT 4", "ChatGPT 4.1", "ChatGPT 4o"}:
        # OpenAI Chat API
        import httpx
        system_prompt = """You are an expert research assistant.{}{}""".format(
            " Do not do any reasoning/thinking, just answer directly." if not think else "",
            " Use all available tools to do deep research." if research else ""
        )
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        model_map = {
            "ChatGPT 4": "gpt-4",
            "ChatGPT 4.1": "gpt-4-1106-preview",
            "ChatGPT 4o": "gpt-4o"
        }
        payload = {
            "model": model_map[model],
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
    elif model in {"Gemini 2.0 Flash Thinking", "Gemini 2.5 Pro"}:
        # Use google-genai SDK for Gemini models
        import asyncio
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                raise RuntimeError("google-generativeai is not installed. Please run: pip install google-generativeai")
        except ImportError:
            raise RuntimeError("google-genai is not installed. Please run: pip install google-generativeai")
        # Map UI model names to Gemini model IDs
        gemini_model_map = {
            "Gemini 2.0 Flash Thinking": "gemini-2.0-flash",
            "Gemini 2.5 Pro": "gemini-2.5.pro-exp-03-25" #models/gemini-2.5-pro-exp-03-25
        }
        model_id = gemini_model_map[model]
        system_prompt = "You are an expert research assistant.{}{}".format(
            " Do not do any reasoning/thinking, just answer directly." if not think else "",
            " Use all available tools to do deep research." if research else ""
        )
        # Configure the client
        genai.configure(api_key=GEMINI_API_KEY)
        # Compose the full prompt
        full_prompt = system_prompt + "\n" + prompt
        model = genai.GenerativeModel(model_id)
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
        # xAI Grok 3 API integration
        import httpx
        system_prompt = "You are an expert research assistant.{}{}".format(
            " Do not do any reasoning/thinking, just answer directly." if not think else "",
            " Use all available tools to do deep research." if research else ""
        )
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "model": "grok-3-latest",
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
    return f"[{model} | Think: {think} | Research: {research}] {prompt} (dummy)"

@app.get("/prompt/stream")
async def prompt_stream(
    prompt: str = Query(...),
    models: List[str] = Query(...),
    leading_model: str = Query(...),
    think: bool = Query(...),
    research: bool = Query(...)
):
    req = PromptRequest(
        prompt=prompt,
        models=models,
        leading_model=leading_model,
        think=think,
        research=research
    )
    return StreamingResponse(stream_model_results(req), media_type="text/event-stream")

async def stream_model_results(req: PromptRequest) -> AsyncGenerator[str, None]:
    async def model_task(model):
        try:
            result = await call_model_api(model, req.prompt, req.think, req.research)
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
    tasks = [call_model_api(m, req.prompt, req.think, req.research) for m in req.models]
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
