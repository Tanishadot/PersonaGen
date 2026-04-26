import pandas as pd
import json
import csv

print("Loading CSV...")
print("Propagating emotions...")
print("Generating pairs...")
print("Saving JSON...")

try:
    # Load training data
    train_path = "data/train.csv"
    print(f"Loading CSV from: {train_path}")
    
    train_df = pd.read_csv(train_path, quoting=csv.QUOTE_ALL, escapechar='\\', on_bad_lines='skip')
    print(f"Train data shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    
    # Extract emotion from self_eval ratings
    def extract_emotion_from_rating(rating):
        """Extract emotion from self_eval rating"""
        if pd.isna(rating):
            return "neutral"
        
        try:
            parts = str(rating).split('|')
            if len(parts) >= 1:
                overall_score = int(parts[0])
                if overall_score >= 5:
                    return "happy"
                elif overall_score >= 4:
                    return "excited"
                elif overall_score >= 3:
                    return "neutral"
                elif overall_score >= 2:
                    return "sad"
                else:
                    return "anxious"
        except:
            return "neutral"
    
    # Apply emotion extraction
    print("Extracting emotions from self_eval ratings...")
    train_df['emotion'] = train_df['selfeval'].apply(extract_emotion_from_rating)
    
    # Extract first non-neutral emotion per conversation
    print("Extracting first non-neutral emotions...")
    conv_emotions = {}
    
    # Group by conversation ID and find first non-neutral emotion
    grouped = train_df.groupby('conv_id')
    for conv_id, group in grouped:
        # Sort by utterance index
        group_sorted = group.sort_values('utterance_idx')
        
        # Find first non-neutral emotion
        for _, row in group_sorted.iterrows():
            emotion = row['emotion']
            if emotion != 'neutral':
                conv_emotions[conv_id] = emotion
                break  # Only store first occurrence
    
    print(f"Found emotions for {len(conv_emotions)} conversations")
    
    # Apply first non-neutral emotions
    train_df['propagated_emotion'] = train_df['conv_id'].map(conv_emotions).fillna('neutral')
    
    # Generate prompt-response pairs
    print("Generating pairs...")
    processed_data = []
    
    # Group by conversation ID
    grouped = train_df.groupby('conv_id')
    
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
                    "emotion": curr_utt['propagated_emotion']
                })
    
    # Limit to 5000 samples
    processed_data = processed_data[:5000]
    
    print(f"Generated {len(processed_data)} prompt-response pairs")
    
    # Validation check
    if len(processed_data) <= 1000:
        raise ValueError(f"Dataset too small: {len(processed_data)} samples. Expected > 1000.")
    
    # Save JSON
    print("Saving JSON...")
    output_path = "data/empathetic_dialogues.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to: {output_path}")
    
    # Show emotion distribution
    emotions = [item['emotion'] for item in processed_data]
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print(f"\nTotal samples: {len(processed_data)}")
    print("Emotion distribution:")
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
    
    print("\nDataset conversion completed successfully!")
    
except Exception as e:
    print(f"ERROR: Failed to convert dataset: {e}")
    raise RuntimeError(f"Dataset conversion failed: {e}")
