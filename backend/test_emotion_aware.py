import requests
import json

def test_emotion_aware_responses():
    """Test emotion-aware response generation"""
    
    base_url = "http://localhost:8000"
    
    test_cases = [
        {
            "message": "I'm feeling really sad today because my friend moved away",
            "persona": "therapist",
            "expected_emotion": "sad"
        },
        {
            "message": "I'm so excited! I just got promoted at work!",
            "persona": "friendly",
            "expected_emotion": "excited"
        },
        {
            "message": "I'm really anxious about the presentation tomorrow",
            "persona": "therapist",
            "expected_emotion": "anxious"
        },
        {
            "message": "I'm so angry at my boss for what he said",
            "persona": "professional",
            "expected_emotion": "angry"
        },
        {
            "message": "Just had a great day at the beach with friends",
            "persona": "friendly",
            "expected_emotion": "happy"
        }
    ]
    
    print("Testing Emotion-Aware Response Generation")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['expected_emotion'].upper()}")
        print(f"Message: {test_case['message']}")
        print(f"Persona: {test_case['persona']}")
        
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "message": test_case["message"],
                    "persona": test_case["persona"],
                    "empathy_threshold": 0.6,
                    "max_retries": 2
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                detected_emotion = result.get('detected_emotion', 'unknown')
                empathy_score = result.get('empathy_score', 0)
                response_text = result.get('response', '')
                
                print(f"Detected Emotion: {detected_emotion}")
                print(f"Empathy Score: {empathy_score:.3f}")
                print(f"Response: {response_text[:100]}...")
                
                # Check if emotion matches expectation
                if detected_emotion == test_case['expected_emotion']:
                    print("Status: PASSED")
                else:
                    print(f"Status: FAILED (expected {test_case['expected_emotion']}, got {detected_emotion})")
                
            else:
                print(f"Status: ERROR - HTTP {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Status: ERROR - {str(e)}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_emotion_aware_responses()
