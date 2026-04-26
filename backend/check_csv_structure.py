import pandas as pd

print("Checking CSV structure...")

# Load a small sample to understand structure
train_path = "data/train.csv"
train_df = pd.read_csv(train_path, quoting=1, escapechar='\\', on_bad_lines='skip', nrows=5)

print("First 5 rows of training data:")
print(train_df[['conv_id', 'utterance_idx', 'context', 'prompt', 'utterance', 'selfeval']].to_string())

print(f"\nColumn names: {list(train_df.columns)}")
print(f"Data types: {train_df.dtypes}")

# Check if there's an emotion column
if 'emotion' in train_df.columns:
    print("Found 'emotion' column!")
    print("Sample emotions:")
    print(train_df['emotion'].head(10))
else:
    print("No 'emotion' column found")
    print("Available columns:", list(train_df.columns))
