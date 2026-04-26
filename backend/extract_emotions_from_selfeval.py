import pandas as pd
import json
import re

print("Extracting emotions from self_eval field...")

def extract_emotion_from_selfeval(selfeval_text):
    """Extract emotion from self_eval text"""
    if pd.isna(selfeval_text):
        return "neutral"
    
    # Convert to string and lowercase
    text = str(selfeval_text).lower()
    
    # Define emotion keywords
    emotion_keywords = {
        'sad': ['sad', 'upset', 'hurt', 'crying', 'tears', 'depressed', 'heartbroken'],
        'happy': ['happy', 'excited', 'glad', 'joy', 'cheerful', 'delighted', 'thrilled'],
        'angry': ['angry', 'mad', 'frustrated', 'annoyed', 'irritated', 'furious'],
        'anxious': ['anxious', 'worried', 'nervous', 'scared', 'afraid', 'fearful'],
        'surprised': ['surprised', 'shocked', 'amazed', 'astonished'],
        'disgusted': ['disgusted', 'revolted', 'repulsed'],
        'fearful': ['fearful', 'terrified', 'horrified', 'scared'],
        'lonely': ['lonely', 'alone', 'isolated'],
        'proud': ['proud', 'accomplished', 'successful'],
        'disappointed': ['disappointed', 'let down', 'failed'],
        'confused': ['confused', 'uncertain', 'unsure'],
        'grateful': ['grateful', 'thankful', 'appreciative'],
        'hopeful': ['hopeful', 'optimistic', 'looking forward'],
        'jealous': ['jealous', 'envious'],
        'embarrassed': ['embarrassed', 'ashamed', 'humiliated']
    }
    
    # Count emotion keywords
    emotion_counts = {}
    for emotion, keywords in emotion_keywords.items():
        count = sum(1 for keyword in keywords if keyword in text)
        if count > 0:
            emotion_counts[emotion] = count
    
    # Return the emotion with highest count
    if emotion_counts:
        return max(emotion_counts, key=emotion_counts.get)
    else:
        return "neutral"

try:
    # Load training data
    train_path = "data/train.csv"
    valid_path = "data/valid.csv"
    test_path = "data/test.csv"
    
    print(f"Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path, quoting=1, escapechar='\\', on_bad_lines='skip')
    print(f"Train data shape: {train_df.shape}")
    
    print(f"Loading validation data from: {valid_path}")
    valid_df = pd.read_csv(valid_path, quoting=1, escapechar='\\', on_bad_lines='skip')
    print(f"Validation data shape: {valid_df.shape}")
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path, quoting=1, escapechar='\\', on_bad_lines='skip')
    print(f"Test data shape: {test_df.shape}")
    
    # Combine all data
    print("Combining datasets...")
    all_data = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    print(f"Combined data shape: {all_data.shape}")
    
    # Extract emotions from self_eval field
    print("Extracting emotions from self_eval...")
    all_data['emotion'] = all_data['selfeval'].apply(extract_emotion_from_selfeval)
    
    print("Sample extracted emotions:")
    for i in range(5):
        selfeval = all_data.iloc[i]['selfeval']
        emotion = all_data.iloc[i]['emotion']
        print(f"  SelfEval: {selfeval} -> Emotion: {emotion}")
    
    # Process data into dialogue pairs
    print("Creating dialogue pairs with extracted emotions...")
    processed_data = []
    
    # Group by conversation ID
    grouped = all_data.groupby('conv_id')
    
    for conv_id, group in grouped:
        # Sort by utterance index
        group_sorted = group.sort_values('utterance_idx')
        
        # Create pairs from consecutive utterances
        for i in range(len(group_sorted) - 1):
            curr_utt = group_sorted.iloc[i]
            next_utt = group_sorted.iloc[i + 1]
            
            # Only include pairs with meaningful content
            if pd.notna(curr_utt['utterance']) and pd.notna(next_utt['utterance']):
                processed_data.append({
                    "context": curr_utt.get('context', ''),
                    "prompt": curr_utt['utterance'],
                    "response": next_utt['utterance'],
                    "emotion": curr_utt['emotion']
                })
    
    # Limit to 5000 samples
    processed_data = processed_data[:5000]
    
    print(f"Processed {len(processed_data)} dialogue pairs")
    
    # Validation check
    if len(processed_data) <= 1000:
        raise ValueError(f"Dataset too small: {len(processed_data)} samples. Expected > 1000.")
    
    # Save as JSON
    output_path = "data/empathetic_dialogues.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"REAL dataset with extracted emotions saved to: {output_path}")
    print(f"Dataset contains {len(processed_data)} real empathetic dialogues")
    
    # Show emotion distribution
    emotions = [item['emotion'] for item in processed_data]
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count} samples")
    
    # Sample verification
    print("\nSample verification:")
    for i in range(min(3, len(processed_data))):
        sample = processed_data[i]
        print(f"  Sample {i+1}:")
        print(f"    Context: {sample['context']}")
        print(f"    Prompt: {sample['prompt'][:80]}...")
        print(f"    Response: {sample['response'][:80]}...")
        print(f"    Emotion: {sample['emotion']}")
        print("---")
    
    print("\n✅ REAL dataset with extracted emotions completed successfully!")
    
except Exception as e:
    print(f"ERROR: Failed to extract emotions: {e}")
    raise RuntimeError(f"Emotion extraction failed: {e}")
