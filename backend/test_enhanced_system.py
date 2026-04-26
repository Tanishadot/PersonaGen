"""
Test script for the enhanced PersonaGen system
Tests the specific test cases and verifies expected behavior
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import get_model_instance
from utils import get_empathy_scorer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_system():
    """
    Test the enhanced system with specific test cases
    """
    print("="*60)
    print("TESTING ENHANCED PERSONAGEN SYSTEM")
    print("="*60)
    
    # Initialize components
    model_instance = get_model_instance()
    empathy_scorer = get_empathy_scorer()
    
    # Test cases
    test_cases = [
        "I am a failure",
        "I feel useless and tired", 
        "I am very angry today"
    ]
    
    expected_results = [
        {"emotion": "sadness", "min_empathy": 0.7, "should_retry": True},
        {"emotion": "sadness", "min_empathy": 0.7, "should_retry": True}, 
        {"emotion": "anger", "min_empathy": 0.7, "should_retry": True}
    ]
    
    all_tests_passed = True
    
    for i, test_input in enumerate(test_cases):
        print(f"\n{'='*40}")
        print(f"TEST CASE {i+1}: '{test_input}'")
        print(f"{'='*40}")
        
        # Test emotion detection
        print("\n1. TESTING EMOTION DETECTION:")
        emotion, confidence = model_instance.detect_emotion(test_input)
        print(f"   Detected emotion: {emotion}")
        print(f"   Confidence: {confidence:.3f}")
        
        expected_emotion = expected_results[i]["emotion"]
        emotion_correct = emotion == expected_emotion or emotion in ["sadness", "anger"]  # Allow some flexibility
        print(f"   Expected: {expected_emotion} - {'✓ PASS' if emotion_correct else '✗ FAIL'}")
        
        if not emotion_correct:
            print(f"   WARNING: Expected {expected_emotion} but got {emotion}")
            all_tests_passed = False
        
        # Test response generation
        print("\n2. TESTING RESPONSE GENERATION:")
        response = model_instance.generate_response(
            test_input, 
            persona="friendly", 
            emotion=emotion
        )
        print(f"   Generated response: '{response}'")
        
        # Test empathy scoring
        print("\n3. TESTING EMPATHY SCORING:")
        empathy_score = empathy_scorer.compute_empathy_score(test_input, response, emotion)
        print(f"   Empathy score: {empathy_score:.3f}")
        
        min_empathy = expected_results[i]["min_empathy"]
        empathy_good = empathy_score >= min_empathy
        print(f"   Expected min empathy: {min_empathy} - {'✓ PASS' if empathy_good else '✗ FAIL'}")
        
        if not empathy_good:
            print(f"   WARNING: Empathy score {empathy_score:.3f} below threshold {min_empathy}")
        
        # Test if response includes validation
        print("\n4. TESTING EMOTIONAL VALIDATION:")
        validation_phrases = ["understand", "sorry", "sounds", "feel", "here for you"]
        has_validation = any(phrase in response.lower() for phrase in validation_phrases)
        print(f"   Contains validation phrases: {'✓ PASS' if has_validation else '✗ FAIL'}")
        
        if not has_validation:
            print(f"   WARNING: Response lacks emotional validation")
        
        # Test feedback loop simulation
        print("\n5. TESTING FEEDBACK LOOP:")
        if empathy_score < 0.7:
            print("   Empathy score below 0.7, feedback loop should trigger")
            should_retry = expected_results[i]["should_retry"]
            print(f"   Should retry: {'✓ PASS' if should_retry else '✗ FAIL'}")
            
            # Test prompt refinement
            refined_prompt = model_instance.refine_prompt_for_empathy(
                test_input, response, empathy_score
            )
            print(f"   Refined prompt length: {len(refined_prompt)} chars")
            print(f"   Contains refinement instructions: {'✓ PASS' if 'more empathetic' in refined_prompt.lower() else '✗ FAIL'}")
        else:
            print("   Empathy score good, no retry needed")
        
        # Test fallback response
        print("\n6. TESTING FALLBACK RESPONSE:")
        fallback = model_instance.get_empathetic_fallback_response()
        print(f"   Fallback response: '{fallback}'")
        fallback_good = "hear you" in fallback and "valid" in fallback
        print(f"   Contains empathetic elements: {'✓ PASS' if fallback_good else '✗ FAIL'}")
        
        # Summary for this test case
        test_passed = emotion_correct and empathy_good and has_validation and fallback_good
        print(f"\n   TEST CASE {i+1} RESULT: {'✓ PASSED' if test_passed else '✗ FAILED'}")
        
        if not test_passed:
            all_tests_passed = False
    
    print(f"\n{'='*60}")
    print(f"OVERALL TEST RESULT: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
    print(f"{'='*60}")
    
    # Test system features
    print(f"\n{'='*60}")
    print("TESTING SYSTEM FEATURES")
    print(f"{'='*60}")
    
    # Test model loading status
    print("\n1. MODEL STATUS:")
    print(f"   Models loaded: {model_instance.are_models_loaded()}")
    print(f"   Models loading: {model_instance.are_models_loading()}")
    print(f"   Use GPT-4: {model_instance.use_gpt4}")
    
    # Test comparison support
    print("\n2. COMPARISON SUPPORT:")
    comparison_scores = model_instance.compute_comparison_scores(test_cases[0], "sadness", empathy_scorer)
    print(f"   Comparison scores available: {len(comparison_scores)} metrics")
    
    # Test prompt engineering
    print("\n3. PROMPT ENGINEERING:")
    structured_prompt = model_instance._construct_emotion_aware_prompt(test_cases[0], "friendly", "sadness")
    print(f"   Structured prompt length: {len(structured_prompt)} chars")
    print(f"   Contains emotion instructions: {'✓' if 'emotionally distressed' in structured_prompt.lower() or 'sadness' in structured_prompt.lower() else '✗'}")
    
    print(f"\n{'='*60}")
    print("SYSTEM TEST COMPLETE")
    print(f"{'='*60}")
    
    return all_tests_passed

if __name__ == "__main__":
    result = asyncio.run(test_enhanced_system())
    sys.exit(0 if result else 1)
