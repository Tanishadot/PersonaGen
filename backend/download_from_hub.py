from huggingface_hub import hf_hub_download
import json

print("Downloading dataset script from HuggingFace Hub...")
try:
    file_path = hf_hub_download(
        repo_id="facebook/empathetic_dialogues",
        filename="empathetic_dialogues.py"
    )
    print(f"Downloaded script to: {file_path}")
    
    # Now we can use the downloaded script to load the dataset
    from datasets import load_dataset
    import importlib.util
    import sys
    import os
    
    # Load the downloaded script
    spec = importlib.util.spec_from_file_location("empathetic_dialogues", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["empathetic_dialogues"] = module
    spec.loader.exec_module(module)
    
    # Load dataset using the script
    dataset = load_dataset("empathetic_dialogues")
    
    print("Processing dataset...")
    processed_data = []
    
    train_data = dataset["train"]
    
    for i in range(len(train_data) - 1):
        curr = train_data[i]
        next_utt = train_data[i + 1]
        
        if curr["conv_id"] == next_utt["conv_id"]:
            processed_data.append({
                "context": curr["context"],
                "prompt": curr["utterance"],
                "response": next_utt["utterance"],
                "emotion": curr["emotion"]
            })
    
    # take only first 5000
    processed_data = processed_data[:5000]
    
    print(f"Total samples: {len(processed_data)}")
    
    with open("data/empathetic_dialogues.json", "w") as f:
        json.dump(processed_data, f, indent=2)
    
    print("Real dataset saved successfully!")
    
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
        }
    ]
    
    with open("data/empathetic_dialogues.json", "w") as f:
        json.dump(fallback_data, f, indent=2)
    
    print("Fallback dataset saved!")
