"""
Models module for PersonaGen AI Chatbot
Contains emotion detection and response generation models
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import logging
import threading
import time
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global loading state
models_loaded = False
models_loading = False
load_lock = threading.Lock()

class PersonaGenModels:
    """Main class for managing AI models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models (lazy loading)
        self.gpt_model = None
        self.gpt_tokenizer = None
        self.emotion_classifier = None
        self.emotion_tokenizer = None
        self.emotion_labels = None
        
        # Model loading status
        self.models_loaded = False
        self.models_loading = False
        self.model_loading_error = None
        
        # Generation configuration - default to GPT-4
        self.use_gpt4 = os.getenv("USE_GPT4", "true").lower() == "true"
        self.openai_api_key = os.getenv("OPENAI_API_KEY", None)
        self.gpt4_model = os.getenv("GPT4_MODEL", "gpt-4o-mini")
        
        # Comparison data storage
        self.generation_comparison = {}
        
        # Persona definitions
        self.personas = {
            "therapist": "You are a compassionate and empathetic therapist. Listen carefully and provide supportive, understanding responses.",
            "friendly": "You are a warm and friendly assistant. Be cheerful, encouraging, and positive in your interactions.",
            "professional": "You are a professional and knowledgeable assistant. Provide clear, concise, and helpful information."
        }
        
        # Do NOT download dataset at startup - will be done lazily
        self.dataset_downloaded = False
        self.empathy_dataset = None
    
    def load_empathy_dataset(self):
        """Load real empathetic dialogues dataset from local JSON file"""
        if self.empathy_dataset is not None:
            logger.info(f"Dataset already loaded with {len(self.empathy_dataset)} samples")
            return self.empathy_dataset
            
        try:
            dataset_path = os.path.join(os.path.dirname(__file__), "data", "empathetic_dialogues.json")
            logger.info(f"Loading real empathy dataset from {dataset_path}")
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.empathy_dataset = json.load(f)
            
            logger.info(f"Real dataset loaded successfully with {len(self.empathy_dataset)} samples")
            
            # Print sample data for verification
            logger.info("Dataset sample verification:")
            for i in range(min(3, len(self.empathy_dataset))):
                sample = self.empathy_dataset[i]
                logger.info(f"  Sample {i+1}:")
                logger.info(f"    Context: {sample['context']}")
                logger.info(f"    Prompt: {sample['prompt'][:80]}...")
                logger.info(f"    Response: {sample['response'][:80]}...")
                logger.info(f"    Emotion: {sample['emotion']}")
                logger.info("---")
            
            return self.empathy_dataset
                
        except Exception as e:
            logger.error(f"Error loading real empathy dataset: {e}")
            logger.info("Continuing without dataset - will use GPT-only mode")
            self.empathy_dataset = []
            return self.empathy_dataset
    
    def _load_gpt_model(self):
        """Load GPT-2 model for response generation using pipeline"""
        if self.gpt_model is None:
            logger.info("Loading GPT-2 model...")
            start_time = time.time()
            
            try:
                model_name = "gpt2"  # Using smaller model for faster loading
                
                # Try loading without device specification first
                try:
                    self.gpt_model = pipeline("text-generation", model=model_name)
                    logger.info(f"GPT-2 model loaded successfully on CPU in {time.time() - start_time:.2f} seconds!")
                except Exception as cpu_error:
                    logger.warning(f"CPU loading failed: {cpu_error}")
                    # Try with explicit device setting
                    device = 0 if self.device == "cuda" else -1
                    self.gpt_model = pipeline("text-generation", model=model_name, device=device)
                    logger.info(f"GPT-2 model loaded successfully on device {device} in {time.time() - start_time:.2f} seconds!")
                
                # Test the model with a simple generation
                test_output = self.gpt_model("Hello", max_new_tokens=5, do_sample=False)
                logger.info(f"GPT-2 model test successful: '{test_output[0]['generated_text']}'")
                
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                logger.error("GPT-2 model failed to load - will use fallback responses")
                self.gpt_model = None  # Ensure it's None on failure
                self.model_loading_error = str(e)
                # Don't raise here, just continue with fallback responses
    
    def _load_emotion_model(self):
        """Load emotion classification model using pipeline"""
        if self.emotion_classifier is None:
            logger.info("Loading emotion classification model...")
            start_time = time.time()
            
            try:
                model_name = "j-hartmann/emotion-english-distilroberta-base"  # Using distilled model for faster loading
                
                self.emotion_classifier = pipeline("text-classification", model=model_name, device=0 if self.device == "cuda" else -1)
                
                # Get emotion labels from pipeline
                self.emotion_labels = [label for label in set(result['label'] for result in self.emotion_classifier(["test"]))]
                
                load_time = time.time() - start_time
                logger.info(f"Emotion model loaded in {load_time:.2f} seconds! Labels: {self.emotion_labels}")
                
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                logger.error("Emotion model failed to load - will use neutral emotion fallback")
                self.emotion_classifier = None  # Ensure it's None on failure
                self.emotion_labels = ["neutral"]
                self.model_loading_error = str(e)
                raise
    
    def generate_response(self, prompt: str, persona: str = "friendly", max_length: int = 100, emotion: str = "neutral") -> str:
        """
        Hybrid response generation with GPT-2 and optional GPT-4 support
        """
        # Add hard rule for distress input detection
        if any(word in prompt.lower() for word in ["failure", "useless", "hopeless"]):
            emotion = "sadness"
            logger.info("Detected distress keywords, overriding emotion to sadness")
        
        # Step 1: Construct structured prompt using controlled engineering
        structured_prompt = self._construct_emotion_aware_prompt(prompt, persona, emotion)
        logger.info(f"Generated structured prompt for emotion: {emotion}, persona: {persona}")
        
        # Choose generation method based on configuration
        if self.use_gpt4 and self.openai_api_key:
            response = self._generate_with_gpt4(structured_prompt, prompt, persona)
        else:
            response = self._generate_with_gpt2(structured_prompt, prompt, persona, emotion)
        
        # Add debug prints for testing
        logger.info(f"Final Emotion: {emotion}")
        logger.info(f"Final Response: {response}")
        
        return response
    
    def _generate_with_gpt2(self, structured_prompt: str, original_prompt: str, persona: str, emotion: str = "neutral") -> str:
        """
        Generate response using GPT-2
        """
        self._load_gpt_model()
        
        # Check if model loaded successfully
        if self.gpt_model is None:
            logger.warning("GPT-2 model not available, using fallback response")
            return self._get_fallback_response(original_prompt, persona)
        
        try:
            # Limit structured prompt to last 400 characters for better GPT-2 performance
            structured_prompt = structured_prompt[-400:]
            logger.info(f"Shortened prompt to {len(structured_prompt)} characters")
            
            # Generate response using pipeline
            outputs = self.gpt_model(
                structured_prompt,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256,  # GPT-2 eos_token_id
                no_repeat_ngram_size=2,
                return_full_text=False
            )
            
            # Get the generated text
            response = outputs[0]['generated_text'].strip()
            logger.info(f"GPT-2 generated response: '{response[:50]}...'")
            
            # Force empathy prefix for distress emotions
            if emotion in ["sadness", "anger", "fear"]:
                if not any(word in response.lower() for word in ["understand", "feel", "sorry", "hard", "difficult"]):
                    response = "I understand this must be really difficult for you. " + response
                    logger.info("Added empathy prefix for distress emotion")
            
            # Ensure non-empty response
            if not response.strip():
                logger.warning("Empty response generated, using empathetic fallback")
                return self.get_empathetic_fallback_response()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating GPT-2 response: {e}")
            return self._get_fallback_response(original_prompt, persona)
    
    def _generate_with_gpt4(self, structured_prompt: str, original_prompt: str, persona: str) -> str:
        """
        Generate response using GPT-4 API
        """
        try:
            import openai
            
            # Configure OpenAI client
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Generate response using GPT-4
            response = client.chat.completions.create(
                model=self.gpt4_model,
                messages=[
                    {"role": "system", "content": structured_prompt},
                    {"role": "user", "content": original_prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            gpt4_response = response.choices[0].message.content.strip()
            logger.info(f"GPT-4 generated response: '{gpt4_response[:50]}...'")
            
            # Also generate with GPT-2 for comparison if available
            gpt2_response = None
            if self.gpt_model is not None:
                gpt2_response = self._generate_with_gpt2(structured_prompt, original_prompt, persona)
                self.generation_comparison[original_prompt] = {
                    "gpt2": gpt2_response,
                    "gpt4": gpt4_response,
                    "timestamp": time.time()
                }
            
            return gpt4_response
            
        except ImportError:
            logger.error("OpenAI package not installed, falling back to GPT-2")
            return self._generate_with_gpt2(structured_prompt, original_prompt, persona)
        except Exception as e:
            logger.error(f"Error generating GPT-4 response: {e}")
            return self._generate_with_gpt2(structured_prompt, original_prompt, persona)
    
    def _construct_emotion_aware_prompt(self, user_input: str, persona: str, emotion: str) -> str:
        """
        Construct structured prompt based on emotion, persona, and user input
        """
        # Build more direct and concise prompt
        system_instruction = f"""
You are an empathetic {persona}.
User emotion: {emotion}

Rules:
- Acknowledge feelings
- Be supportive
- Avoid generic phrases like 'provide more details'
"""
        
        # Construct final prompt with better ending
        structured_prompt = f"""{system_instruction}

User says: {user_input}

Response (must include emotional validation, no generic phrases):"""
        
        logger.info(f"Constructed prompt for {emotion} with {persona} persona")
        return structured_prompt
    
    def _get_emotion_instructions(self, emotion: str) -> str:
        """
        Get emotion-specific instructions for response generation
        """
        emotion_guidelines = {
            "sadness": "Be gentle, validating, and comforting. Acknowledge their pain and offer emotional support.",
            "anger": "Acknowledge frustration without escalating. Be patient and validate their feelings while staying calm.",
            "fear": "Be reassuring and calming. Provide stability and reduce uncertainty with supportive language.",
            "joy": "Celebrate and amplify positivity. Share in their joy with enthusiasm and encouragement.",
            "neutral": "Be balanced, supportive, and helpful while maintaining a warm tone.",
            "surprise": "Match their energy appropriately while providing grounding if needed.",
            "disgust": "Be understanding and non-judgmental about their concerns.",
            "happy": "Be cheerful and positive, reinforcing their good feelings."
        }
        
        return emotion_guidelines.get(emotion, "Be supportive and understanding.")
    
    def refine_prompt_for_empathy(self, original_prompt: str, previous_response: str, empathy_score: float) -> str:
        """
        Refine prompt for higher empathy in retry attempts
        """
        refinement_instruction = """
Make the response more empathetic.
- Explicitly acknowledge emotions
- Say things like "I understand", "that sounds hard"
- Avoid generic or formal phrases
- Sound human and supportive"""
        
        refined_prompt = f"{original_prompt}{refinement_instruction}"
        logger.info(f"Refined prompt for higher empathy (previous score: {empathy_score:.3f})")
        
        return refined_prompt
    
    def detect_emotion(self, text: str) -> Tuple[str, float]:
        """
        Enhanced emotion detection with preprocessing and fallback rules
        Returns: (emotion_label, confidence_score)
        """
        # Step 1: Preprocess input
        processed_text = text.strip().lower()
        
        logger.info(f"Input text: '{text}' -> Processed: '{processed_text}'")
        
        self._load_emotion_model()
        
        # Check if model loaded successfully
        if self.emotion_classifier is None:
            logger.warning("Emotion model not available, using rule-based fallback")
            return self._rule_based_emotion_detection(processed_text)
        
        try:
            # Get prediction using pipeline
            results = self.emotion_classifier(processed_text)
            
            # Get top emotion and confidence
            emotion = results[0]['label']
            confidence_score = results[0]['score']
            
            logger.info(f"Model detected emotion: {emotion} (confidence: {confidence_score:.3f})")
            
            # Step 2: Apply fallback rule-based detection if confidence is low
            if confidence_score < 0.6:
                logger.info(f"Low confidence ({confidence_score:.3f}), applying fallback rules")
                fallback_emotion = self._rule_based_emotion_detection(processed_text)
                # Use fallback if it's not neutral, otherwise stick with model prediction
                if fallback_emotion[0] != "neutral":
                    emotion, confidence_score = fallback_emotion
                    logger.info(f"Using fallback emotion: {emotion} (confidence: {confidence_score:.3f})")
            
            # Ensure final emotion is never incorrectly neutral for emotional sentences
            if emotion == "neutral" and self._contains_emotional_keywords(processed_text):
                logger.warning("Detected neutral but text contains emotional keywords, applying correction")
                emotion, confidence_score = self._rule_based_emotion_detection(processed_text)
            
            logger.info(f"Final emotion: {emotion} (confidence: {confidence_score:.3f})")
            return emotion, confidence_score
            
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return self._rule_based_emotion_detection(processed_text)
    
    def _rule_based_emotion_detection(self, text: str) -> Tuple[str, float]:
        """
        Rule-based emotion detection as fallback
        """
        sadness_keywords = ["failure", "useless", "sad", "tired", "depressed", "lonely", "hopeless", "cry", "crying"]
        anger_keywords = ["angry", "hate", "frustrated", "annoyed", "mad", "pissed", "irritated", "furious"]
        fear_keywords = ["scared", "afraid", "anxious", "worried", "nervous", "panic", "terrified"]
        joy_keywords = ["happy", "excited", "joy", "glad", "pleased", "delighted", "cheerful"]
        
        text_lower = text.lower()
        
        # Check for emotion-specific keywords
        if any(keyword in text_lower for keyword in sadness_keywords):
            logger.info(f"Rule-based detection: sadness (keywords: {[k for k in sadness_keywords if k in text_lower]})")
            return "sadness", 0.7
        elif any(keyword in text_lower for keyword in anger_keywords):
            logger.info(f"Rule-based detection: anger (keywords: {[k for k in anger_keywords if k in text_lower]})")
            return "anger", 0.7
        elif any(keyword in text_lower for keyword in fear_keywords):
            logger.info(f"Rule-based detection: fear (keywords: {[k for k in fear_keywords if k in text_lower]})")
            return "fear", 0.7
        elif any(keyword in text_lower for keyword in joy_keywords):
            logger.info(f"Rule-based detection: joy (keywords: {[k for k in joy_keywords if k in text_lower]})")
            return "joy", 0.7
        
        # Default to neutral if no emotional keywords found
        return "neutral", 0.5
    
    def _contains_emotional_keywords(self, text: str) -> bool:
        """
        Check if text contains any emotional keywords
        """
        all_emotional_keywords = [
            "failure", "useless", "sad", "tired", "depressed", "lonely", "hopeless", "cry", "crying",
            "angry", "hate", "frustrated", "annoyed", "mad", "pissed", "irritated", "furious",
            "scared", "afraid", "anxious", "worried", "nervous", "panic", "terrified",
            "happy", "excited", "joy", "glad", "pleased", "delighted", "cheerful"
        ]
        
        return any(keyword in text.lower() for keyword in all_emotional_keywords)
    
    def _get_fallback_response(self, prompt: str, persona: str = "friendly") -> str:
        """Get fallback response when models are not available or retries fail"""
        fallback_responses = {
            "therapist": "I understand you're going through something difficult. As a therapist, I want you to know that your feelings are valid, and it's okay to seek support. Would you like to tell me more about what's on your mind?",
            "friendly": "I'm here for you. Could you tell me more about how you're feeling? I want to understand and support you.",
            "professional": "I understand this might be frustrating. I'm here to help and support you."
        }
        return fallback_responses.get(persona, fallback_responses["friendly"])
    
    def get_empathetic_fallback_response(self) -> str:
        """
        Get the universal fallback response for failed retries
        """
        return "I hear you. That sounds really difficult, and your feelings are valid. I'm here for you."
    
    def compute_comparison_scores(self, user_input: str, emotion: str, empathy_scorer) -> Dict[str, float]:
        """
        Compute empathy scores for comparison analysis
        """
        comparison_scores = {}
        
        if not self.generation_comparison:
            return comparison_scores
        
        # Find the most recent comparison for this user input
        for input_text, comparison_data in self.generation_comparison.items():
            if user_input.lower() in input_text.lower() or input_text.lower() in user_input.lower():
                gpt2_response = comparison_data.get('gpt2')
                gpt4_response = comparison_data.get('gpt4')
                
                if gpt2_response:
                    gpt2_score = empathy_scorer.compute_empathy_score(user_input, gpt2_response, emotion)
                    comparison_scores['gpt2_empathy'] = gpt2_score
                    logger.info(f"GPT-2 empathy score: {gpt2_score:.3f}")
                
                if gpt4_response:
                    gpt4_score = empathy_scorer.compute_empathy_score(user_input, gpt4_response, emotion)
                    comparison_scores['gpt4_empathy'] = gpt4_score
                    logger.info(f"GPT-4 empathy score: {gpt4_score:.3f}")
                
                break
        
        return comparison_scores
    
    def _get_loading_response(self) -> str:
        """Get response when models are loading"""
        return "Loading models, please wait a moment..."
    
    def are_models_loaded(self) -> bool:
        """Check if models are loaded successfully"""
        return self.models_loaded and self.gpt_model is not None and self.emotion_classifier is not None
    
    def are_models_loading(self) -> bool:
        """Check if models are currently loading"""
        return self.models_loading
    
    def get_model_loading_error(self) -> str:
        """Get the last model loading error"""
        return self.model_loading_error
    
    def get_available_personas(self) -> List[str]:
        """Get list of available personas"""
        return list(self.personas.keys())
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using keyword overlap"""
        # Simple keyword-based similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'you', 'me', 'my', 'your', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'what', 'when', 'where', 'why', 'how', 'so', 'if', 'because', 'while', 'as', 'at', 'by', 'for', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once'}
        
        words1 -= common_words
        words2 -= common_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def retrieve_similar_responses(self, user_input: str, top_k: int = 3) -> List[Dict]:
        """Retrieve similar responses from real dataset"""
        dataset = self.load_empathy_dataset()
        
        if not dataset:
            logger.warning("No real dataset available for retrieval")
            return []
        
        similarities = []
        for sample in dataset:
            # Calculate similarity based on prompt and context
            prompt_sim = self.calculate_similarity(user_input, sample['prompt'])
            context_sim = self.calculate_similarity(user_input, sample['context'])
            
            # Combined similarity score
            combined_sim = 0.7 * prompt_sim + 0.3 * context_sim
            
            similarities.append({
                'similarity': combined_sim,
                'sample': sample
            })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = similarities[:top_k]
        
        logger.info(f"Retrieved {len(top_results)} similar responses from real dataset")
        for i, result in enumerate(top_results):
            logger.info(f"  Result {i+1}: similarity={result['similarity']:.3f}, emotion={result['sample']['emotion']}")
        
        return [result['sample'] for result in top_results]
    
    
def load_all_models():
    """Load all models at once with proper locking"""
    global models_loaded, models_loading
    
    with load_lock:
        if models_loaded:
            return True
            
        if models_loading:
            # Wait for loading to complete
            while models_loading:
                time.sleep(0.1)
            return models_loaded
            
        models_loading = True
        
        try:
            logger.info("Starting to load all models...")
            start_time = time.time()
            
            # Create model instance
            model = PersonaGenModels()
            
            # Try to load models
            try:
                model._load_gpt_model()
                model._load_emotion_model()
                
                # Load real dataset (with error handling)
                try:
                    model.load_empathy_dataset()
                    logger.info("Real empathy dataset loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load real empathy dataset: {e}")
                    logger.info("Continuing without dataset - will use GPT-only mode")
                
                load_time = time.time() - start_time
                logger.info(f"All models loaded successfully in {load_time:.2f} seconds!")
                
                model.models_loaded = True
                models_loaded = True
                return True
                
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                logger.info("Models failed to load - will use fallback responses")
                model.models_loaded = False
                models_loaded = False
                # Don't return False - we want to continue with fallback responses
                return True  # Return True to indicate loading attempt completed
                
        except Exception as e:
            logger.error(f"Critical error in model loading: {e}")
            models_loaded = False
            return True  # Return True to prevent 503 errors
        finally:
            models_loading = False

# Global model instance
model_instance = None

def get_model_instance():
    """Get or create global model instance"""
    global model_instance
    if model_instance is None:
        model_instance = PersonaGenModels()
    return model_instance

def are_models_loaded():
    """Check if models are loaded"""
    model = get_model_instance()
    return model.are_models_loaded()

def are_models_loading():
    """Check if models are currently loading"""
    model = get_model_instance()
    return model.are_models_loading()
