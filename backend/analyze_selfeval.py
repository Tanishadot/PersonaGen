import pandas as pd

print("Analyzing self_eval field patterns...")

# Load a sample to understand self_eval patterns
train_path = "data/train.csv"
train_df = pd.read_csv(train_path, quoting=1, escapechar='\\', on_bad_lines='skip', nrows=20)

print("Sample self_eval values:")
for i, row in train_df.iterrows():
    selfeval = row['selfeval']
    print(f"  {i+1}: {selfeval}")

print(f"\nUnique self_eval values (first 20):")
unique_values = train_df['selfeval'].dropna().unique()[:20]
for val in unique_values:
    print(f"  {val}")

print(f"\nTotal unique self_eval values: {len(train_df['selfeval'].dropna().unique())}")

# Look for patterns in the data
print("\nLooking for emotion patterns...")
for i, row in train_df.iterrows():
    context = str(row['context']).lower()
    if any(word in context for word in ['sad', 'upset', 'hurt', 'crying']):
        print(f"  Row {i+1}: Context suggests sadness - {context[:50]}...")
    elif any(word in context for word in ['happy', 'excited', 'glad', 'joy']):
        print(f"  Row {i+1}: Context suggests happiness - {context[:50]}...")
    elif any(word in context for word in ['angry', 'mad', 'frustrated']):
        print(f"  Row {i+1}: Context suggests anger - {context[:50]}...")
