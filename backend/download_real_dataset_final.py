from datasets import load_dataset
import json

print("Downloading REAL EmpatheticDialogues dataset from HuggingFace...")

# Use streaming mode to avoid script issues
try:
    dataset = load_dataset("empathetic_dialogues", split="train", streaming=True)
    
    print("Processing dataset from stream...")
    processed_data = []
    
    # Collect 5000 samples
    for i, sample in enumerate(dataset):
        if i >= 5000:
            break
            
        # Create dialogue pairs (current and next utterance)
        if i < 4999:  # Ensure we have next sample
            curr_sample = sample
            next_sample = next(iter(dataset))
            
            # Only create pairs from same conversation
            if curr_sample.get("conv_id") == next_sample.get("conv_id"):
                processed_data.append({
                    "context": curr_sample.get("context", ""),
                    "prompt": curr_sample.get("utterance", ""),
                    "response": next_sample.get("utterance", ""),
                    "emotion": curr_sample.get("emotion", "")
                })
        
        # Progress logging
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} samples...")
    
    print(f"Collected {len(processed_data)} samples")
    
    # Validation check - must have > 1000 samples
    if len(processed_data) <= 1000:
        raise ValueError(f"Dataset too small: {len(processed_data)} samples. Expected > 1000.")
    
    # Save dataset locally
    with open("data/empathetic_dialogues.json", "w", encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print("REAL dataset saved successfully!")
    print(f"Dataset contains {len(processed_data)} real empathetic dialogues")
    
    # Print sample verification
    print("\nSample verification:")
    for i in range(min(3, len(processed_data))):
        sample = processed_data[i]
        print(f"  Sample {i+1}:")
        print(f"    Context: {sample['context']}")
        print(f"    Prompt: {sample['prompt'][:80]}...")
        print(f"    Response: {sample['response'][:80]}...")
        print(f"    Emotion: {sample['emotion']}")
        print("---")
    
except Exception as e:
    print(f"ERROR: Failed to download real dataset: {e}")
    print("FAIL-SAFE: System requires real dataset to function.")
    print("No fallback dataset will be created.")
    raise RuntimeError(f"Real dataset download failed: {e}")
