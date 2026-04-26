import pandas as pd
import json
import csv

print("Fixing emotion labels in EmpatheticDialogues dataset...")

try:
    # Load training data
    train_path = "data/train.csv"
    valid_path = "data/valid.csv"
    test_path = "data/test.csv"
    
    print(f"Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path, quoting=csv.QUOTE_ALL, escapechar='\\', on_bad_lines='skip')
    print(f"Train data shape: {train_df.shape}")
    
    print(f"Loading validation data from: {valid_path}")
    valid_df = pd.read_csv(valid_path, quoting=csv.QUOTE_ALL, escapechar='\\', on_bad_lines='skip')
    print(f"Validation data shape: {valid_df.shape}")
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path, quoting=csv.QUOTE_ALL, escapechar='\\', on_bad_lines='skip')
    print(f"Test data shape: {test_df.shape}")
    
    # Combine all data
    print("Combining datasets...")
    all_data = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    print(f"Combined data shape: {all_data.shape}")
    
    # Track conversation emotions
    print("Tracking conversation-level emotions...")
    conv_emotions = {}
    
    # First pass: find the actual emotion for each conversation
    for _, row in all_data.iterrows():
        conv_id = row['conv_id']
        emotion = row.get('emotion', 'neutral')
        
        # Only store non-neutral emotions
        if conv_id not in conv_emotions and emotion != 'neutral':
            conv_emotions[conv_id] = emotion
            print(f"  Conversation {conv_id}: {emotion}")
    
    print(f"Found emotions for {len(conv_emotions)} conversations")
    
    # Process data into dialogue pairs
    print("Creating dialogue pairs with correct emotions...")
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
                # Use conversation-level emotion
                emotion = conv_emotions.get(conv_id, 'neutral')
                
                processed_data.append({
                    "context": curr_utt.get('context', ''),
                    "prompt": curr_utt['utterance'],
                    "response": next_utt['utterance'],
                    "emotion": emotion
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
    
    print(f"Fixed dataset saved to: {output_path}")
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
    
    print("\n✅ Emotion labeling fixed successfully!")
    
except Exception as e:
    print(f"ERROR: Failed to fix emotion labels: {e}")
    raise RuntimeError(f"Emotion labeling failed: {e}")
