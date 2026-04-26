from datasets import load_dataset
import json

print("Downloading dataset...")
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

with open("backend/data/empathetic_dialogues.json", "w") as f:
    json.dump(processed_data, f, indent=2)

print("Dataset saved successfully!")