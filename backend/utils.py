"""
Utils module for PersonaGen AI Chatbot
Contains empathy scoring, evaluation metrics, and memory management
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import logging
import math
import threading
from collections import Counter, OrderedDict

logger = logging.getLogger(__name__)

def word_tokenize(text: str) -> List[str]:
    """Lightweight tokenizer (words and punctuation)."""
    return re.findall(r"\w+|[^\w\s]", text)


def sentence_bleu(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """Sentence BLEU with add-one smoothing (equivalent in spirit to NLTK's method1)."""
    if not hypothesis or not reference:
        return 0.0
    log_precision = 0.0
    for n in range(1, max_n + 1):
        hyp = Counter(tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1))
        ref = Counter(tuple(reference[i:i + n]) for i in range(len(reference) - n + 1))
        total = sum(hyp.values())
        overlap = sum(min(c, ref[g]) for g, c in hyp.items())
        if total == 0:
            precision = 1.0 / (len(hypothesis) + 1)
        elif overlap == 0:
            precision = 0.1 / total
        else:
            precision = overlap / total
        log_precision += math.log(precision) / max_n
    bp = 1.0 if len(hypothesis) > len(reference) else math.exp(1 - len(reference) / len(hypothesis))
    return bp * math.exp(log_precision)


class EmpathyScorer:
    """Class for computing empathy scores"""
    
    def __init__(self):
        # Empathy indicators
        self.empathy_words = [
            'understand', 'feel', 'sorry', 'care', 'support', 'listen',
            'empathize', 'compassion', 'difficult', 'hard', 'struggle',
            'experience', 'going through', 'must be', 'sounds like',
            'imagine', 'appreciate', 'validate', 'acknowledge'
        ]
        
        self.supportive_words = [
            'help', 'here for you', 'together', 'guidance', 'advice',
            'suggestions', 'resources', 'options', 'solutions', 'assist'
        ]
        
        self.emotional_words = [
            'sad', 'happy', 'angry', 'frustrated', 'anxious', 'worried',
            'excited', 'proud', 'disappointed', 'confused', 'overwhelmed'
        ]
    
    def compute_empathy_score(self, user_input: str, response: str, detected_emotion: str) -> float:
        """
        Enhanced empathy score with weighted system and penalties
        Returns score between 0.0 and 1.0
        """
        logger.info(f"Computing empathy score for response: '{response[:50]}...'")
        
        # Weighted scoring system
        score = 0.0
        
        # Factor 1: empathy_words (0.4)
        empathy_words_score = self._calculate_empathy_language_score(response)
        score += empathy_words_score * 0.4
        logger.info(f"Empathy words score: {empathy_words_score:.3f} (weighted: {empathy_words_score * 0.4:.3f})")
        
        # Factor 2: emotional_acknowledgment (0.3)
        emotion_ack_score = self._calculate_emotion_acknowledgment_score(user_input, response, detected_emotion)
        score += emotion_ack_score * 0.3
        logger.info(f"Emotional acknowledgment score: {emotion_ack_score:.3f} (weighted: {emotion_ack_score * 0.3:.3f})")
        
        # Factor 3: supportive_language (0.2)
        supportive_lang_score = self._calculate_support_score(response)
        score += supportive_lang_score * 0.2
        logger.info(f"Supportive language score: {supportive_lang_score:.3f} (weighted: {supportive_lang_score * 0.2:.3f})")
        
        # Factor 4: response_length (0.1)
        length_score = self._calculate_length_score(response)
        score += length_score * 0.1
        logger.info(f"Response length score: {length_score:.3f} (weighted: {length_score * 0.1:.3f})")
        
        # Apply penalties
        initial_score = score
        
        # Penalty 1: Technical/unhelpful phrases (-0.4)
        technical_phrases = ["provide more details", "technical issue", "cannot assist", "unable to help", "not available"]
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in technical_phrases):
            score -= 0.4
            logger.warning(f"Technical phrase detected, applying -0.4 penalty")
        
        # Penalty 2: No emotional acknowledgment (-0.3)
        acknowledgment_phrases = ["i understand", "that sounds", "i'm sorry", "that must be", "i can see", "it sounds like"]
        if not any(phrase in response_lower for phrase in acknowledgment_phrases):
            score -= 0.3
            logger.warning(f"No emotional acknowledgment detected, applying -0.3 penalty")
        
        # Ensure score is clamped between 0 and 1
        final_score = max(0.0, min(1.0, score))
        
        logger.info(f"Initial score: {initial_score:.3f}, Final score: {final_score:.3f}")
        
        return final_score
    
    def _calculate_empathy_language_score(self, response: str) -> float:
        """Calculate score based on empathy-related words"""
        response_lower = response.lower()
        empathy_count = sum(1 for word in self.empathy_words if word in response_lower)
        
        # Normalize by response length
        words = len(response.split())
        if words == 0:
            return 0.0
        
        # Score based on density of empathy words
        density = empathy_count / words
        return min(density * 10, 1.0)  # Cap at 1.0
    
    def _calculate_emotion_acknowledgment_score(self, user_input: str, response: str, detected_emotion: str) -> float:
        """Calculate score for acknowledging user's emotions"""
        response_lower = response.lower()
        user_lower = user_input.lower()
        
        score = 0.0
        
        # Check if response mentions the detected emotion
        if detected_emotion.lower() in response_lower:
            score += 0.5
        
        # Check if response uses emotional language from user input
        user_emotional_words = [word for word in self.emotional_words if word in user_lower]
        if user_emotional_words:
            response_emotional_matches = sum(1 for word in user_emotional_words if word in response_lower)
            score += (response_emotional_matches / len(user_emotional_words)) * 0.5
        
        return min(score, 1.0)
    
    def _calculate_length_score(self, response: str) -> float:
        """Calculate score based on appropriate response length"""
        word_count = len(response.split())
        
        # Ideal range is 10-50 words
        if 10 <= word_count <= 50:
            return 1.0
        elif word_count < 10:
            return word_count / 10.0
        else:
            return max(0.5, 1.0 - (word_count - 50) / 100.0)
    
    def _calculate_support_score(self, response: str) -> float:
        """Calculate score based on supportive language"""
        response_lower = response.lower()
        support_count = sum(1 for word in self.supportive_words if word in response_lower)
        
        words = len(response.split())
        if words == 0:
            return 0.0
        
        density = support_count / words
        return min(density * 15, 1.0)
    
    def _calculate_personalization_score(self, user_input: str, response: str) -> float:
        """Calculate score based on personalized response"""
        # Check if response references user input
        user_words = set(word.lower() for word in word_tokenize(user_input.lower()))
        response_words = set(word.lower() for word in word_tokenize(response.lower()))
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        user_words -= common_words
        response_words -= common_words
        
        # Calculate overlap
        if not user_words:
            return 0.0
        
        overlap = len(user_words.intersection(response_words))
        return min(overlap / len(user_words), 1.0)

class EvaluationMetrics:
    """Class for computing evaluation metrics"""
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score for response quality"""
        try:
            return sentence_bleu(word_tokenize(reference.lower()), word_tokenize(hypothesis.lower()))
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_response_diversity(self, responses: List[str]) -> float:
        """Calculate diversity of responses (avoid repetition)"""
        if len(responses) < 2:
            return 1.0
        
        # Calculate pairwise similarity
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self._calculate_jaccard_similarity(responses[i], responses[j])
                total_similarity += similarity
                comparisons += 1
        
        if comparisons == 0:
            return 1.0
        
        avg_similarity = total_similarity / comparisons
        # Diversity is inverse of average similarity
        return 1.0 - avg_similarity
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(word.lower() for word in word_tokenize(text1.lower()))
        words2 = set(word.lower() for word in word_tokenize(text2.lower()))
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 1.0
        
        return intersection / union

class MemoryManager:
    """Class for managing conversation memory"""
    
    def __init__(self, max_memory_size: int = 5):
        self.memory = deque(maxlen=max_memory_size)
        self.max_memory_size = max_memory_size
    
    def add_message(self, user_input: str, bot_response: str, emotion: str, empathy_score: float):
        """Add a message pair to memory"""
        message = {
            'user_input': user_input,
            'bot_response': bot_response,
            'emotion': emotion,
            'empathy_score': empathy_score,
            'timestamp': self._get_timestamp()
        }
        self.memory.append(message)
    
    def get_context(self) -> str:
        """Get formatted context from memory"""
        if not self.memory:
            return ""
        
        context_parts = []
        for i, msg in enumerate(self.memory):
            context_parts.append(f"Previous conversation {i+1}:")
            context_parts.append(f"User: {msg['user_input']}")
            context_parts.append(f"Assistant: {msg['bot_response']}")
            context_parts.append(f"Detected emotion: {msg['emotion']}, Empathy score: {msg['empathy_score']:.2f}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def get_history(self) -> List[Dict]:
        """Previous turns, oldest first, as [{"user": ..., "assistant": ...}]"""
        return [{"user": m['user_input'], "assistant": m['bot_response']} for m in self.memory]

    def get_recent_emotions(self, count: int = 3) -> List[str]:
        """Get recent emotions from memory"""
        recent_messages = list(self.memory)[-count:]
        return [msg['emotion'] for msg in recent_messages]
    
    def get_average_empathy_score(self) -> float:
        """Get average empathy score from memory"""
        if not self.memory:
            return 0.0
        
        scores = [msg['empathy_score'] for msg in self.memory]
        return np.mean(scores)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class SessionStore:
    """One MemoryManager per browser session, so visitors don't share a conversation."""

    def __init__(self, max_sessions: int = 1000, max_memory_size: int = 5):
        self._sessions: "OrderedDict[str, MemoryManager]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.max_memory_size = max_memory_size

    def get(self, session_id: str) -> MemoryManager:
        session_id = (session_id or "default")[:64]
        with self._lock:
            memory = self._sessions.pop(session_id, None) or MemoryManager(self.max_memory_size)
            self._sessions[session_id] = memory  # most recently used last
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return memory


# Global instances
empathy_scorer = EmpathyScorer()
evaluator = EvaluationMetrics()
session_store = SessionStore()
memory_manager = session_store.get("default")

def get_empathy_scorer():
    """Get global empathy scorer instance"""
    return empathy_scorer

def get_evaluator():
    """Get global evaluator instance"""
    return evaluator

def get_memory_manager(session_id: str = "default"):
    """Get the memory manager for a session"""
    return session_store.get(session_id)
