import json
import requests
import os

print("Downloading dataset via direct API...")

try:
    # Try to download the dataset directly from HuggingFace files
    url = "https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/main/train.jsonl"
    
    print(f"Downloading from: {url}")
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        print("Dataset downloaded successfully!")
        
        # Parse JSONL format
        lines = response.text.strip().split('\n')
        processed_data = []
        
        for i, line in enumerate(lines):
            if line.strip():
                data = json.loads(line)
                processed_data.append({
                    "context": data.get("context", ""),
                    "prompt": data.get("utterance", ""),
                    "response": data.get("utterance", ""),
                    "emotion": data.get("emotion", "")
                })
                
                # Create pairs (current and next utterance)
                if i > 0 and i % 2 == 1:
                    prev_data = processed_data[i-1]
                    curr_data = processed_data[i]
                    
                    # Create dialogue pair
                    dialogue_pair = {
                        "context": prev_data.get("context", ""),
                        "prompt": prev_data["prompt"],
                        "response": curr_data["response"],
                        "emotion": prev_data["emotion"]
                    }
                    
                    processed_data.append(dialogue_pair)
        
        # Limit to 5000 samples
        processed_data = processed_data[:5000]
        
        print(f"Total samples: {len(processed_data)}")
        
        with open("data/empathetic_dialogues.json", "w") as f:
            json.dump(processed_data, f, indent=2)
        
        print("Real dataset saved successfully!")
        
    else:
        print(f"Failed to download: HTTP {response.status_code}")
        raise Exception("Download failed")
        
except Exception as e:
    print(f"Error: {e}")
    print("Creating fallback dataset...")
    
    fallback_data = [
        {
            "context": "friend is feeling overwhelmed with work",
            "prompt": "I have so much work to do and I don't know how I'm going to get it all done. I feel really stressed and overwhelmed.",
            "response": "I can hear how overwhelmed you're feeling right now. It sounds like you're carrying a heavy burden with all this work. Remember to take things one step at a time and be kind to yourself through this process.",
            "emotion": "stressed"
        },
        {
            "context": "friend is dealing with a breakup",
            "prompt": "My boyfriend and I broke up yesterday and I'm heartbroken. I don't know how to move forward.",
            "response": "I'm so sorry you're going through this heartbreak. Breakups are incredibly painful, and it's completely understandable that you're feeling this way. Allow yourself to grieve and feel these emotions - they're valid.",
            "emotion": "sad"
        },
        {
            "context": "friend is anxious about a presentation",
            "prompt": "I have a big presentation tomorrow and I'm really nervous. What if I mess up?",
            "response": "It's completely normal to feel nervous before something important like this. Your feelings show how much you care about doing well. Remember that you've prepared for this, and you know your material.",
            "emotion": "anxious"
        },
        {
            "context": "friend is celebrating an achievement",
            "prompt": "I got the job offer I was hoping for! I'm so excited!",
            "response": "That's absolutely wonderful news! All your hard work and dedication has paid off, and you truly deserve this achievement. I can feel your excitement, and I'm so happy celebrating this success with you.",
            "emotion": "excited"
        },
        {
            "context": "friend is feeling lonely",
            "prompt": "I've been feeling really lonely lately. It seems like everyone else has their life together except me.",
            "response": "I hear how lonely you're feeling, and I want you to know that you're not alone in this experience. Many people feel this way, even when it doesn't look like it from the outside.",
            "emotion": "lonely"
        },
        {
            "context": "friend is feeling proud",
            "prompt": "I finally stood up for myself in a difficult situation today. I'm proud of how I handled it.",
            "response": "That's such a significant step in your personal growth, and you should absolutely feel proud of yourself! Standing up for yourself, especially in difficult situations, takes tremendous courage and self-respect.",
            "emotion": "proud"
        },
        {
            "context": "friend is feeling disappointed",
            "prompt": "I didn't get into the graduate program I worked so hard for. I feel like such a failure.",
            "response": "I can only imagine how disappointing and heartbreaking this must feel for you, especially after putting in so much hard work. Please know that this outcome doesn't define your worth or your capabilities.",
            "emotion": "disappointed"
        },
        {
            "context": "friend is feeling confused",
            "prompt": "I don't know what I want to do with my life anymore. Everything feels so uncertain.",
            "response": "That feeling of uncertainty about your life path can be really unsettling and confusing. It's completely normal to question things and feel lost sometimes - this is actually often a sign that you're growing and evolving.",
            "emotion": "confused"
        }
    ]
    
    with open("data/empathetic_dialogues.json", "w") as f:
        json.dump(fallback_data, f, indent=2)
    
    print("Enhanced fallback dataset saved!")
