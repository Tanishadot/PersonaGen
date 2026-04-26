from datasets import load_dataset
import json

print("Downloading dataset...")
try:
    dataset = load_dataset("facebook/empathetic_dialogues")
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
    
    print("Dataset saved successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print("Creating fallback dataset...")
    
    fallback_data = [
        {
            "context": "friend is feeling overwhelmed with work",
            "prompt": "I have so much work to do and I don't know how I'm going to get it all done. I feel really stressed and overwhelmed.",
            "response": "I can hear how overwhelmed you're feeling right now. It sounds like you're carrying a heavy burden with all this work. Remember to take things one step at a time and be kind to yourself through this process.",
            "emotion": "stressed"
        }
    ]
    
    with open("data/empathetic_dialogues.json", "w") as f:
        json.dump(fallback_data, f, indent=2)
    
    print("Fallback dataset saved!")
