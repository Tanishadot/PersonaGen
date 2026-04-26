import pandas as pd
import csv

print("Analyzing emotion distribution in original data...")

# Load training data
train_path = "data/train.csv"
train_df = pd.read_csv(train_path, quoting=csv.QUOTE_ALL, escapechar='\\', on_bad_lines='skip')

print(f"Training data shape: {train_df.shape}")

# Extract emotion from self_eval ratings
def extract_emotion_from_rating(rating):
    """Extract emotion from self_eval rating"""
    if pd.isna(rating):
        return "neutral"
    
    try:
        parts = str(rating).split('|')
        if len(parts) >= 1:
            overall_score = int(parts[0])
            return overall_score
    except:
        return 0

# Apply emotion extraction
train_df['emotion_score'] = train_df['selfeval'].apply(extract_emotion_from_rating)

print("\nEmotion score distribution (first 1000 rows):")
score_counts = {}
for score in train_df['emotion_score'].head(1000):
    score_counts[score] = score_counts.get(score, 0) + 1

for score in sorted(score_counts.keys()):
    print(f"  Score {score}: {score_counts[score]} samples")

# Map scores to emotions
def map_score_to_emotion(score):
    """Map empathy score to emotion with better distribution"""
    try:
        score = int(score)
        if score >= 5:
            return "happy"
        elif score >= 4:
            return "excited"
        elif score >= 3:
            return "neutral"
        elif score >= 2:
            return "sad"
        else:
            return "anxious"
    except:
        return "neutral"

# Apply better emotion mapping
train_df['emotion'] = train_df['emotion_score'].apply(map_score_to_emotion)

print("\nBetter emotion distribution (first 1000 rows):")
emotion_counts = {}
for emotion in train_df['emotion'].head(1000):
    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

for emotion in sorted(emotion_counts.keys()):
    print(f"  {emotion}: {emotion_counts[emotion]} samples")

# Check context-based emotions
print("\nAnalyzing context-based emotions...")
context_emotions = {'sad': 0, 'happy': 0, 'angry': 0, 'anxious': 0, 'excited': 0}

for _, row in train_df.head(500).iterrows():
    context = str(row['context']).lower()
    if any(word in context for word in ['sad', 'upset', 'hurt', 'crying']):
        context_emotions['sad'] += 1
    elif any(word in context for word in ['happy', 'excited', 'glad', 'joy']):
        context_emotions['happy'] += 1
    elif any(word in context for word in ['angry', 'mad', 'frustrated']):
        context_emotions['angry'] += 1
    elif any(word in context for word in ['worried', 'anxious', 'nervous']):
        context_emotions['anxious'] += 1
    elif any(word in context for word in ['excited', 'thrilled']):
        context_emotions['excited'] += 1

print("Context-based emotion distribution:")
for emotion, count in context_emotions.items():
    print(f"  {emotion}: {count} samples")
