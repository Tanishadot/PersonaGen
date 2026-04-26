from transformers import pipeline

print("Loading GPT...")
gen = pipeline("text-generation", model="gpt2")

print(gen("I feel sad")[0]["generated_text"])

print("Loading RoBERTa...")
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

print(classifier("I feel anxious"))
