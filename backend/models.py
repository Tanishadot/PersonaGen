"""
Models module for PersonaGen AI Chatbot

- Emotion detection: j-hartmann/emotion-english-distilroberta-base
  (labels: anger, disgust, fear, joy, neutral, sadness, surprise)
- Response generation: OpenAI chat model when OPENAI_API_KEY is set,
  otherwise local GPT-2
- Retrieval: TF-IDF search over EmpatheticDialogues (see prepare_dataset.py)
"""

import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import pipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "empathetic_dialogues.json"),
)
EMOTION_MODEL = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
GPT2_MODEL = os.getenv("GPT2_MODEL", "gpt2")

PERSONAS = {
    "therapist": "You are a compassionate, empathetic therapist. Listen carefully and respond with "
                 "warmth and understanding. Ask gentle open questions.",
    "friendly": "You are a warm, friendly companion. Be cheerful, encouraging and supportive.",
    "professional": "You are a professional, knowledgeable assistant. Be clear, concise and helpful "
                    "while staying kind.",
}

# Keyed by the classifier's labels so the detected emotion always finds its guideline.
EMOTION_GUIDELINES = {
    "sadness": "be gentle, validating and comforting; acknowledge their pain before anything else",
    "fear": "be reassuring and calming; provide stability and reduce uncertainty",
    "anger": "acknowledge their frustration, validate it, stay patient and avoid escalating",
    "joy": "celebrate with them and share in their happiness; be enthusiastic",
    "surprise": "match their energy and help them make sense of what happened",
    "disgust": "be understanding and non-judgmental about what bothered them",
    "neutral": "be balanced, supportive, clear and helpful",
}

REFINEMENT_INSTRUCTION = (
    "Your previous reply was not empathetic enough. Explicitly acknowledge how they feel "
    "(for example \"I understand\", \"that sounds really hard\"), avoid generic or formal "
    "phrases, and sound human and supportive."
)


class PersonaGenModels:
    """Loads and runs the emotion classifier, the generator and the retriever."""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.gpt_model = None
        self.emotion_classifier = None

        self.empathy_dataset: List[Dict] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._doc_matrix = None

        self.models_loaded = False
        self.models_loading = False
        self.model_loading_error: Optional[str] = None
        self._lock = threading.Lock()

        self.openai_api_key = os.getenv("OPENAI_API_KEY") or None
        self.openai_model = os.getenv("OPENAI_MODEL", os.getenv("GPT4_MODEL", "gpt-4o-mini"))
        self.use_openai = bool(self.openai_api_key) and os.getenv("USE_GPT4", "true").lower() == "true"
        self._openai_client = None

        self.personas = PERSONAS

    # ------------------------------------------------------------------ loading

    def load(self):
        """Load everything once. Safe to call from several threads."""
        with self._lock:
            if self.models_loaded:
                return
            self.models_loading = True
            start = time.time()
            try:
                self._load_emotion_model()
                # GPT-2 is only needed when we are not using the OpenAI API.
                if not self.use_openai or os.getenv("LOAD_GPT2", "false").lower() == "true":
                    self._load_gpt_model()
                self.load_empathy_dataset()
                logger.info(f"Models ready in {time.time() - start:.1f}s "
                            f"(generator: {'openai:' + self.openai_model if self.use_openai else 'gpt2'})")
            except Exception as e:  # keep serving with fallbacks
                logger.exception("Model loading failed")
                self.model_loading_error = str(e)
            finally:
                self.models_loaded = True
                self.models_loading = False

    def _load_gpt_model(self):
        if self.gpt_model is None:
            logger.info(f"Loading {GPT2_MODEL} ...")
            try:
                self.gpt_model = pipeline("text-generation", model=GPT2_MODEL, device=self.device)
            except Exception as e:
                logger.error(f"GPT-2 failed to load: {e}")
                self.model_loading_error = str(e)

    def _load_emotion_model(self):
        if self.emotion_classifier is None:
            logger.info(f"Loading {EMOTION_MODEL} ...")
            try:
                self.emotion_classifier = pipeline("text-classification", model=EMOTION_MODEL, device=self.device)
            except Exception as e:
                logger.error(f"Emotion model failed to load, using keyword rules: {e}")
                self.model_loading_error = str(e)

    def load_empathy_dataset(self) -> List[Dict]:
        """Load EmpatheticDialogues pairs and build a TF-IDF index for retrieval."""
        if self.empathy_dataset:
            return self.empathy_dataset
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                self.empathy_dataset = json.load(f)
        except FileNotFoundError:
            logger.warning(f"{DATASET_PATH} not found - run `python prepare_dataset.py`. Retrieval disabled.")
            self.empathy_dataset = []
            return self.empathy_dataset

        docs = [f"{s['prompt']} {s['situation']}" for s in self.empathy_dataset]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self._doc_matrix = self._vectorizer.fit_transform(docs)
        logger.info(f"Indexed {len(self.empathy_dataset)} dialogue pairs for retrieval")
        return self.empathy_dataset

    def are_models_loaded(self) -> bool:
        return self.models_loaded

    def are_models_loading(self) -> bool:
        return self.models_loading

    def generator_name(self) -> str:
        if self.use_openai:
            return f"openai:{self.openai_model}"
        return "gpt2" if self.gpt_model is not None else "fallback"

    # ------------------------------------------------------------------ emotion

    def detect_emotion(self, text: str) -> Tuple[str, float]:
        processed = text.strip().lower()
        if self.emotion_classifier is None:
            return self._rule_based_emotion_detection(processed)
        try:
            result = self.emotion_classifier(processed, truncation=True)[0]
            emotion, confidence = result["label"].lower(), float(result["score"])
            if confidence < 0.6 or (emotion == "neutral" and self._contains_emotional_keywords(processed)):
                rule_emotion, rule_conf = self._rule_based_emotion_detection(processed)
                if rule_emotion != "neutral":
                    emotion, confidence = rule_emotion, rule_conf
            return emotion, confidence
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return self._rule_based_emotion_detection(processed)

    _KEYWORDS = {
        "sadness": ["failure", "useless", "sad", "tired", "depressed", "lonely", "hopeless", "cry", "crying"],
        "anger": ["angry", "hate", "frustrated", "annoyed", "mad", "pissed", "irritated", "furious"],
        "fear": ["scared", "afraid", "anxious", "worried", "nervous", "panic", "terrified"],
        "joy": ["happy", "excited", "joy", "glad", "pleased", "delighted", "cheerful"],
    }

    def _rule_based_emotion_detection(self, text: str) -> Tuple[str, float]:
        for emotion, words in self._KEYWORDS.items():
            if any(w in text for w in words):
                return emotion, 0.7
        return "neutral", 0.5

    def _contains_emotional_keywords(self, text: str) -> bool:
        return any(w in text for words in self._KEYWORDS.values() for w in words)

    # ---------------------------------------------------------------- retrieval

    def retrieve_similar_responses(self, user_input: str, top_k: int = 3,
                                   emotion: Optional[str] = None, min_similarity: float = 0.05) -> List[Dict]:
        """
        Return up to top_k dataset pairs most similar to user_input, each with a
        `similarity` field. If `emotion` is given, pairs with that emotion are
        preferred among the 50 nearest candidates.
        """
        if not self.empathy_dataset or self._vectorizer is None:
            return []
        query = self._vectorizer.transform([user_input])
        scores = linear_kernel(query, self._doc_matrix).ravel()
        candidates = scores.argsort()[::-1][:50]
        candidates = [i for i in candidates if scores[i] >= min_similarity]
        if emotion:
            matching = [i for i in candidates if self.empathy_dataset[i]["emotion"] == emotion]
            if matching:
                candidates = matching
        return [{**self.empathy_dataset[i], "similarity": float(scores[i])} for i in candidates[:top_k]]

    # --------------------------------------------------------------- generation

    def build_system_prompt(self, persona: str, emotion: str,
                            example: Optional[Dict] = None, refine: bool = False) -> str:
        """Emotion-aware system prompt: persona + tone guideline + retrieved example."""
        parts = [
            PERSONAS.get(persona, PERSONAS["friendly"]),
            f"The user is currently feeling: {emotion}.",
            f"Tone guidelines: {EMOTION_GUIDELINES.get(emotion, EMOTION_GUIDELINES['neutral'])}.",
            "Respond with emotional understanding and validation, then offer comfort or helpful "
            "guidance. Avoid generic phrases like 'provide more details'. Keep it natural, "
            "conversational and under 120 words.",
        ]
        if example:
            parts.append(
                "Here is how an empathetic listener replied to someone in a similar situation. "
                "Use it as inspiration for tone, do not copy it:\n"
                f"Person: {example['prompt']}\nListener: {example['response']}"
            )
        if refine:
            parts.append(REFINEMENT_INSTRUCTION)
        return "\n\n".join(parts)

    def generate_response(self, user_input: str, persona: str = "friendly", emotion: str = "neutral",
                          history: Optional[List[Dict]] = None, example: Optional[Dict] = None,
                          refine: bool = False) -> str:
        """
        history: previous turns, oldest first, as [{"user": ..., "assistant": ...}]
        example: a retrieved dataset pair used as a tone example
        """
        history = history or []
        system_prompt = self.build_system_prompt(persona, emotion, example, refine)
        if self.use_openai:
            response = self._generate_with_openai(system_prompt, user_input, history)
            if response:
                return response
            logger.warning("OpenAI generation failed, falling back to GPT-2")
            self._load_gpt_model()
        return self._generate_with_gpt2(user_input, persona, emotion, history, example)

    def _generate_with_openai(self, system_prompt: str, user_input: str, history: List[Dict]) -> Optional[str]:
        try:
            if self._openai_client is None:
                import openai
                self._openai_client = openai.OpenAI(api_key=self.openai_api_key, timeout=30)
            messages = [{"role": "system", "content": system_prompt}]
            for turn in history:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
            messages.append({"role": "user", "content": user_input})
            completion = self._openai_client.chat.completions.create(
                model=self.openai_model, messages=messages, max_tokens=250, temperature=0.7,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None

    def _generate_with_gpt2(self, user_input: str, persona: str, emotion: str,
                            history: List[Dict], example: Optional[Dict]) -> str:
        if self.gpt_model is None:
            return self._get_fallback_response(user_input, persona)

        # GPT-2 is a plain completion model, so give it a short dialogue transcript:
        # the retrieved pair as a one-shot example, then recent history, then the user.
        lines = [f"A conversation with a caring {persona} who {EMOTION_GUIDELINES.get(emotion, 'is supportive')}.", ""]
        if example:
            lines += [f"User: {example['prompt']}", f"Assistant: {example['response']}"]
        for turn in history[-2:]:
            lines += [f"User: {turn['user']}", f"Assistant: {turn['assistant']}"]
        lines += [f"User: {user_input}", "Assistant:"]
        prompt = "\n".join(lines)

        tokenizer = self.gpt_model.tokenizer
        ids = tokenizer.encode(prompt)
        if len(ids) > 800:  # GPT-2 context is 1024 tokens; leave room for the reply
            prompt = tokenizer.decode(ids[-800:])

        try:
            out = self.gpt_model(
                prompt, max_new_tokens=80, temperature=0.7, do_sample=True, top_p=0.9,
                no_repeat_ngram_size=3, pad_token_id=tokenizer.eos_token_id, return_full_text=False,
            )
            text = out[0]["generated_text"]
            # Stop at the next turn GPT-2 invents for the user.
            text = text.strip().split("\n")[0]
            for stop in ("User:", "Assistant:", "User and"):
                text = text.split(stop)[0]
            response = text.strip()
            if emotion in ("sadness", "anger", "fear") and not any(
                    w in response.lower() for w in ("understand", "feel", "sorry", "hard", "difficult")):
                response = "I understand this must be really difficult for you. " + response
            return response or self.get_empathetic_fallback_response()
        except Exception as e:
            logger.error(f"GPT-2 generation failed: {e}")
            return self._get_fallback_response(user_input, persona)

    # ---------------------------------------------------------------- fallbacks

    def _get_fallback_response(self, prompt: str, persona: str = "friendly") -> str:
        fallback = {
            "therapist": "I understand you're going through something difficult. Your feelings are valid, "
                         "and it's okay to seek support. Would you like to tell me more about what's on your mind?",
            "friendly": "I'm here for you. Could you tell me more about how you're feeling? "
                        "I want to understand and support you.",
            "professional": "I understand this might be frustrating. I'm here to help and support you.",
        }
        return fallback.get(persona, fallback["friendly"])

    def get_empathetic_fallback_response(self) -> str:
        return "I hear you. That sounds really difficult, and your feelings are valid. I'm here for you."

    def _get_loading_response(self) -> str:
        return "I'm still waking up (loading models). Please try again in a few seconds."

    def get_available_personas(self) -> List[str]:
        return list(PERSONAS.keys())


# ---------------------------------------------------------------- module API

_model_instance: Optional[PersonaGenModels] = None
_instance_lock = threading.Lock()


def get_model_instance() -> PersonaGenModels:
    global _model_instance
    with _instance_lock:
        if _model_instance is None:
            _model_instance = PersonaGenModels()
        return _model_instance


def load_all_models() -> bool:
    """Load models into the shared instance (blocking)."""
    get_model_instance().load()
    return True


def start_background_loading():
    """Kick off model loading without blocking server startup."""
    model = get_model_instance()
    if not model.models_loaded and not model.models_loading:
        model.models_loading = True  # so /health reports "loading" immediately
        threading.Thread(target=model.load, daemon=True).start()


def are_models_loaded() -> bool:
    return get_model_instance().are_models_loaded()


def are_models_loading() -> bool:
    return get_model_instance().are_models_loading()
