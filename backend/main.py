"""
FastAPI Backend for PersonaGen: Emotionally Intelligent Chatbot
Main application with API endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import time
import asyncio

from models import get_model_instance, load_all_models, are_models_loaded, are_models_loading
from utils import get_empathy_scorer, get_evaluator, get_memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PersonaGen API",
    description="Emotionally Intelligent Chatbot with Feedback Loop",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    persona: Optional[str] = "friendly"
    empathy_threshold: Optional[float] = 0.6
    max_retries: Optional[int] = 3

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

# Global variables
start_time = time.time()
model_instance = None
empathy_scorer = None
evaluator = None
memory_manager = None

async def initialize_utils():
    """Initialize utility modules"""
    global empathy_scorer, evaluator, memory_manager
    try:
        logger.info("Initializing utility modules...")
        empathy_scorer = get_empathy_scorer()
        evaluator = get_evaluator()
        memory_manager = get_memory_manager()
        logger.info("Utility modules initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing utility modules: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize utility modules only - models will be loaded lazily"""
    await initialize_utils()
    logger.info("PersonaGen API started successfully! Models will load on first request.")

def get_emotion_aware_prompt(user_input: str, emotion: str, persona: str) -> str:
    """
    Generate emotion-aware prompt for GPT response generation
    """
    # Emotion-specific tone guidelines
    emotion_guidelines = {
        "sad": "be gentle, validating, comforting, acknowledge their pain",
        "anxious": "be reassuring, calming, provide stability, reduce uncertainty",
        "angry": "acknowledge frustration, validate feelings, avoid escalation, be patient",
        "happy": "celebrate and amplify positivity, share in their joy, be enthusiastic",
        "excited": "match their energy, be enthusiastic, build on their excitement",
        "neutral": "be balanced, supportive, clear and helpful",
        "lonely": "be warm, connecting, remind them they're not alone, offer companionship",
        "disappointed": "validate their feelings, offer perspective, be encouraging",
        "proud": "celebrate their achievement, acknowledge their effort, share in success",
        "confused": "be patient, clarify, offer structure, reduce complexity"
    }
    
    # Get specific guidelines for detected emotion
    guidelines = emotion_guidelines.get(emotion, "be supportive and understanding")
    
    prompt = f"""You are an empathetic AI assistant.

The user is feeling: {emotion}

Tone guidelines: {guidelines}

User says: {user_input}

Respond with:
- emotional understanding and validation
- supportive tone following the guidelines above
- helpful guidance or comfort
- avoid being overly generic

Make your response natural and conversational."""
    
    return prompt

def blend_responses(gpt_response: str, retrieved_response: str, empathy_score: float) -> str:
    """
    Blend GPT response with retrieved response based on empathy score
    """
    if empathy_score >= 0.7:
        # High empathy: use mostly GPT response
        return gpt_response
    elif empathy_score >= 0.5:
        # Medium empathy: blend responses
        # Take first part from GPT, second part from retrieved
        gpt_words = gpt_response.split()
        retrieved_words = retrieved_response.split()
        
        if len(gpt_words) > 20 and len(retrieved_words) > 20:
            # Blend at sentence boundary
            gpt_half = len(gpt_words) // 2
            blended = ' '.join(gpt_words[:gpt_half]) + ' ' + ' '.join(retrieved_words[gpt_half//2:])
            return blended
        else:
            return gpt_response
    else:
        # Low empathy: use retrieved response
        return retrieved_response

def generate_response_with_feedback(
    user_input: str,
    persona: str,
    empathy_threshold: float,
    max_retries: int
) -> Dict[str, Any]:
    """
    Generate response with empathy feedback loop and retrieval enhancement
    """
    # Get model instance
    model_instance = get_model_instance()
    
    # Load models if not already loaded (never fail)
    if not model_instance.are_models_loaded():
        if model_instance.are_models_loading():
            logger.info("Models are currently loading, using loading response...")
            return {
                "response": model_instance._get_loading_response(),
                "detected_emotion": "neutral",
                "emotion_confidence": 0.5,
                "empathy_score": 0.5,
                "retry_count": 0,
                "persona": persona,
                "context_used": False,
                "bleu_score": 0.0,
                "retrieved_context": None,
                "retrieval_used": False
            }
        else:
            logger.info("Models not loaded, attempting to load...")
            load_all_models()  # This will always return True now
            
            # Check if models loaded successfully after attempt
            if not model_instance.are_models_loaded():
                logger.warning("Models failed to load, using fallback response")
                return {
                    "response": model_instance._get_fallback_response(user_input, persona),
                    "detected_emotion": "neutral",
                    "emotion_confidence": 0.5,
                    "empathy_score": 0.5,
                    "retry_count": 0,
                    "persona": persona,
                    "context_used": False,
                    "bleu_score": 0.0,
                    "retrieved_context": None,
                    "retrieval_used": False
                }
    
    retry_count = 0
    best_response = ""
    best_empathy_score = 0.0
    detected_emotion = "neutral"
    emotion_confidence = 0.5
    retrieved_context = ""
    retrieval_used = False
    
    try:
        # Get conversation context
        context = memory_manager.get_context()
        context_used = bool(context)
        
        # Add context to user input if available
        if context:
            enhanced_input = f"Context: {context}\n\nCurrent user message: {user_input}"
        else:
            enhanced_input = user_input
        
        # Detect emotion first to guide response generation
        logger.info(f"Detecting emotion in user input...")
        emotion, confidence = model_instance.detect_emotion(user_input)
        logger.info(f"Detected emotion: {emotion} (confidence: {confidence:.3f})")
        detected_emotion = emotion
        emotion_confidence = confidence
        
        # Step 1: Retrieve similar responses from dataset
        logger.info("Retrieving similar responses from dataset...")
        retrieved_responses = model_instance.retrieve_similar_responses(user_input, top_k=3)
        
        # Filter retrieved responses by detected emotion
        emotion_filtered_responses = []
        if retrieved_responses:
            for resp in retrieved_responses:
                if resp['emotion'] == emotion:
                    emotion_filtered_responses.append(resp)
            
            # If we have emotion-matching responses, use those
            if emotion_filtered_responses:
                retrieved_responses = emotion_filtered_responses
                logger.info(f"Using {len(emotion_filtered_responses)} emotion-matching responses")
            else:
                logger.info(f"No emotion-matching responses found, using top {len(retrieved_responses)} responses")
            
            retrieved_context = retrieved_responses[0]['response']
            retrieval_used = True
            logger.info(f"Retrieved similar response with emotion: {retrieved_responses[0]['emotion']}")
        
        # Step 2: Generate emotion-aware prompt
        emotion_aware_prompt = get_emotion_aware_prompt(user_input, emotion, persona)
        logger.info(f"Generated emotion-aware prompt for {emotion}")
        
        # Implement proper feedback loop
        best_response = None
        best_score = 0.0
        current_prompt = user_input
        
        for i in range(max_retries):
            try:
                # Generate response
                response = model_instance.generate_response(
                    current_prompt,
                    persona=persona,
                    emotion=detected_emotion
                )
                
                # Compute empathy score
                score = empathy_scorer.compute_empathy_score(
                    user_input,
                    response,
                    detected_emotion
                )
                
                # Debug log
                print(f"Retry {i}: Score = {score}")
                logger.info(f"Retry {i}: Score = {score}")
                
                # Update best response
                if score > best_score:
                    best_score = score
                    best_response = response
                    logger.info(f"New best score: {best_score}")
                
                # Check if threshold met
                if score >= empathy_threshold:
                    logger.info(f"Threshold met: {score} >= {empathy_threshold}")
                    break
                
                # Refine prompt for next iteration
                current_prompt = model_instance.refine_prompt_for_empathy(
                    user_input,
                    response,
                    score
                )
                logger.info(f"Refined prompt for next retry")
                
            except Exception as e:
                logger.error(f"Error in retry {i}: {e}")
                continue
        
        # Update final values
        best_empathy_score = best_score
        retry_count = i
        
        # Handle failure case
        if best_score < empathy_threshold:
            logger.warning(f"Best score {best_score} below threshold {empathy_threshold}, using empathetic fallback")
            best_response = model_instance.get_empathetic_fallback_response()
            best_empathy_score = 0.5
        
        # Return proper metrics matching ChatResponse format
        return {
            "response": best_response,
            "detected_emotion": detected_emotion,
            "emotion_confidence": emotion_confidence,
            "empathy_score": best_score,
            "retry_count": i,
            "persona": persona,
            "context_used": context_used,
            "bleu_score": None,
            "retrieved_context": None,
            "retrieval_used": False
        }
        
    except Exception as e:
        logger.error(f"Critical error in response generation: {e}")
        # Always return a valid response, never fail
        return {
            "response": model_instance._get_fallback_response(user_input, persona),
            "detected_emotion": "neutral",
            "emotion_confidence": 0.5,
            "empathy_score": 0.5,
            "retry_count": max_retries,
            "persona": persona,
            "context_used": False,
            "bleu_score": 0.0,
            "retrieved_context": None,
            "retrieval_used": False
        }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - never returns 503 errors
    """
    if not request.message.strip():
        # Return a proper response instead of 400 error
        return ChatResponse(
            response="Please provide a message so I can help you.",
            detected_emotion="neutral",
            emotion_confidence=0.5,
            empathy_score=0.5,
            retry_count=0,
            persona=request.persona,
            context_used=False,
            bleu_score=0.0,
            retrieved_context=None,
            retrieval_used=False
        )
    
    try:
        result = generate_response_with_feedback(
            request.message,
            request.persona,
            request.empathy_threshold,
            request.max_retries
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        # Always return a valid response, never fail
        model_instance = get_model_instance()
        return ChatResponse(
            response=model_instance._get_fallback_response(request.message, request.persona),
            detected_emotion="neutral",
            emotion_confidence=0.5,
            empathy_score=0.5,
            retry_count=0,
            persona=request.persona,
            context_used=False,
            bleu_score=0.0,
            retrieved_context=None,
            retrieval_used=False
        )

@app.get("/personas", response_model=PersonaResponse)
async def get_personas():
    """
    Get available personas
    """
    # Personas are static, so we can return them without loading models
    personas = ["therapist", "friendly", "professional"]
    return PersonaResponse(personas=personas)

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get conversation metrics - never returns 503 errors
    """
    try:
        if not memory_manager:
            # Return default metrics instead of 503 error
            return MetricsResponse(
                average_empathy_score=0.0,
                total_conversations=0,
                recent_emotions=[],
                response_diversity=1.0
            )
        
        avg_empathy = memory_manager.get_average_empathy_score()
        recent_emotions = memory_manager.get_recent_emotions(5)
        
        # Calculate response diversity
        responses = [msg['bot_response'] for msg in memory_manager.memory]
        diversity = evaluator.calculate_response_diversity(responses) if responses else 1.0
        
        return MetricsResponse(
            average_empathy_score=avg_empathy,
            total_conversations=len(memory_manager.memory),
            recent_emotions=recent_emotions,
            response_diversity=diversity
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        # Return default metrics on any error
        return MetricsResponse(
            average_empathy_score=0.0,
            total_conversations=0,
            recent_emotions=[],
            response_diversity=1.0
        )

@app.delete("/memory")
async def clear_memory():
    """
    Clear conversation memory - never returns 503 errors
    """
    try:
        if not memory_manager:
            return {"message": "Memory manager not available, but continuing..."}
        
        memory_manager.clear_memory()
        return {"message": "Memory cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return {"message": "Memory cleared with some issues, but continuing..."}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    uptime = time.time() - start_time
    models_loaded_status = are_models_loaded()
    models_loading_status = are_models_loading()
    
    if models_loading_status:
        status = "loading"
    elif models_loaded_status:
        status = "healthy"
    else:
        status = "ready"
    
    return HealthResponse(
        status=status,
        models_loaded=models_loaded_status,
        uptime=uptime
    )

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "PersonaGen API - Emotionally Intelligent Chatbot",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "personas": "/personas",
            "metrics": "/metrics",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
