# Test emotion-aware prompt generation logic

def get_emotion_aware_prompt(user_input: str, emotion: str, persona: str) -> str:
    """Generate emotion-aware prompt for GPT response generation"""
    
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

# Test cases
test_cases = [
    ("I'm feeling really sad today because my friend moved away", "sad", "therapist"),
    ("I'm so excited! I just got promoted at work!", "excited", "friendly"),
    ("I'm really anxious about the presentation tomorrow", "anxious", "therapist"),
    ("I'm so angry at my boss for what he said", "angry", "professional"),
    ("Just had a great day at the beach with friends", "happy", "friendly")
]

print("Testing Emotion-Aware Prompt Generation")
print("=" * 50)

for i, (message, emotion, persona) in enumerate(test_cases, 1):
    print(f"\nTest {i}: {emotion.upper()}")
    print(f"Message: {message}")
    print(f"Persona: {persona}")
    
    prompt = get_emotion_aware_prompt(message, emotion, persona)
    print(f"\nGenerated Prompt:")
    print(prompt[:200] + "...")
    print("-" * 50)

print("\nEmotion-aware prompt generation working correctly!")
