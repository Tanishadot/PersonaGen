"""
FastAPI Backend for PersonaGen: Emotionally Intelligent Chatbot

Pipeline per message:
  1. detect the user's emotion
  2. retrieve a similar EmpatheticDialogues pair (filtered by emotion)
  3. build an emotion-aware prompt (persona + tone guideline + retrieved example)
     and include the session's conversation history
  4. generate, score empathy, and regenerate with a refined prompt if below threshold
  5. store the turn in session memory
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models import get_model_instance, load_all_models, start_background_loading
from utils import get_empathy_scorer, get_evaluator, get_memory_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PersonaGen API",
    description="Emotionally Intelligent Chatbot with Feedback Loop",
    version="1.1.0",
)

# In production the frontend is served by this same app, so CORS only matters
# for local development (React dev server on :3000) or a separately hosted UI.
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    persona: Optional[str] = "friendly"
    empathy_threshold: Optional[float] = 0.6
    max_retries: Optional[int] = 3
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    detected_emotion: str
    emotion_confidence: float
    empathy_score: float
    retry_count: int
    persona: str
    context_used: bool
    bleu_score: Optional[float] = None
    retrieved_context: Optional[str] = None
    retrieval_used: bool = False
    generator: Optional[str] = None


class PersonaResponse(BaseModel):
    personas: List[str]


class MetricsResponse(BaseModel):
    average_empathy_score: float
    total_conversations: int
    recent_emotions: List[str]
    response_diversity: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    uptime: float
    generator: Optional[str] = None


start_time = time.time()
empathy_scorer = get_empathy_scorer()
evaluator = get_evaluator()


@app.on_event("startup")
async def startup_event():
    # Load models in the background so the server (and health checks) come up immediately.
    if os.getenv("PRELOAD_MODELS", "true").lower() == "true":
        start_background_loading()
    logger.info("PersonaGen API started")


def _fallback_result(text: str, persona: str, retry_count: int = 0) -> Dict[str, Any]:
    return {
        "response": text,
        "detected_emotion": "neutral",
        "emotion_confidence": 0.5,
        "empathy_score": 0.5,
        "retry_count": retry_count,
        "persona": persona,
        "context_used": False,
        "bleu_score": None,
        "retrieved_context": None,
        "retrieval_used": False,
        "generator": "fallback",
    }


def generate_response_with_feedback(
    user_input: str,
    persona: str,
    empathy_threshold: float,
    max_retries: int,
    session_id: str = "default",
) -> Dict[str, Any]:
    model = get_model_instance()

    if model.are_models_loading():
        return _fallback_result(model._get_loading_response(), persona)
    if not model.are_models_loaded():
        load_all_models()  # PRELOAD_MODELS=false: load on first request

    memory = get_memory_manager(session_id)

    try:
        # 1. Conversation memory
        history = memory.get_history()
        context_used = bool(history)

        # 2. Emotion
        emotion, confidence = model.detect_emotion(user_input)
        logger.info(f"Detected emotion: {emotion} ({confidence:.2f})")

        # 3. Retrieval (prefers pairs with the same emotion)
        # Include the previous user message so short follow-ups keep their topic.
        query = f"{history[-1]['user']} {user_input}" if history else user_input
        retrieved = model.retrieve_similar_responses(query, top_k=3, emotion=emotion)
        example = retrieved[0] if retrieved else None

        # 4. Generate -> score -> refine loop
        best_response, best_score, attempts = "", -1.0, 0
        for attempt in range(max(1, max_retries)):
            attempts = attempt + 1
            response = model.generate_response(
                user_input,
                persona=persona,
                emotion=emotion,
                history=history,
                example=example,
                refine=attempt > 0,
            )
            score = empathy_scorer.compute_empathy_score(user_input, response, emotion)
            logger.info(f"Attempt {attempts}: empathy={score:.3f}")
            if response and score > best_score:
                best_response, best_score = response, score
            if score >= empathy_threshold:
                break

        if not best_response:
            best_response = model.get_empathetic_fallback_response()
            best_score = empathy_scorer.compute_empathy_score(user_input, best_response, emotion)

        bleu = evaluator.calculate_bleu_score(example["response"], best_response) if example else None

        # 5. Remember this turn
        memory.add_message(user_input, best_response, emotion, best_score)

        return {
            "response": best_response,
            "detected_emotion": emotion,
            "emotion_confidence": confidence,
            "empathy_score": best_score,
            "retry_count": attempts - 1,
            "persona": persona,
            "context_used": context_used,
            "bleu_score": bleu,
            "retrieved_context": example["response"] if example else None,
            "retrieval_used": example is not None,
            "generator": model.generator_name(),
        }
    except Exception:
        logger.exception("Response generation failed")
        return _fallback_result(model._get_fallback_response(user_input, persona), persona, max_retries)


# Plain `def` (not async) so FastAPI runs the blocking model calls in a worker
# thread instead of freezing the event loop for every other request.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    persona = request.persona if request.persona in get_model_instance().personas else "friendly"
    if not request.message.strip():
        return ChatResponse(**_fallback_result("Please provide a message so I can help you.", persona))

    threshold = min(max(request.empathy_threshold or 0.6, 0.0), 1.0)
    retries = min(max(request.max_retries or 3, 1), 5)
    result = generate_response_with_feedback(
        request.message.strip(), persona, threshold, retries, request.session_id or "default"
    )
    return ChatResponse(**result)


@app.get("/personas", response_model=PersonaResponse)
async def get_personas():
    return PersonaResponse(personas=get_model_instance().get_available_personas())


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(session_id: str = "default"):
    memory = get_memory_manager(session_id)
    responses = [m["bot_response"] for m in memory.memory]
    return MetricsResponse(
        average_empathy_score=float(memory.get_average_empathy_score()),
        total_conversations=len(memory.memory),
        recent_emotions=memory.get_recent_emotions(5),
        response_diversity=evaluator.calculate_response_diversity(responses) if responses else 1.0,
    )


@app.delete("/memory")
async def clear_memory(session_id: str = "default"):
    get_memory_manager(session_id).clear_memory()
    return {"message": "Memory cleared successfully"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    model = get_model_instance()
    if model.are_models_loading():
        status = "loading"
    elif model.are_models_loaded():
        status = "healthy"
    else:
        status = "ready"  # models load on first request
    return HealthResponse(
        status=status,
        models_loaded=model.are_models_loaded(),
        uptime=time.time() - start_time,
        generator=model.generator_name() if model.are_models_loaded() else None,
    )


@app.get("/api")
async def api_info():
    return {
        "message": "PersonaGen API - Emotionally Intelligent Chatbot",
        "version": app.version,
        "endpoints": ["/chat", "/personas", "/metrics", "/memory", "/health", "/docs"],
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "status_code": exc.status_code})


# Serve the built React app (frontend/build) from the same server when present.
FRONTEND_DIR = os.getenv(
    "FRONTEND_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "build"),
)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return await api_info()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
