import pandas as pd
import os

# Change working directory to the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working directory set to:", os.getcwd())

# File paths
music_info_path = 'data/Music Info.csv'
listening_history_path = 'data/User Listening History.csv'
output_path = 'data/cleaned_merged_data.csv'

# Step 1: Load Music Info dataset
music_df = pd.read_csv(music_info_path)
print("\nMusic Info Dataset Loaded. First few rows:")
print(music_df.head())

# Step 2: Clean Music Info dataset
music_df['genre'] = music_df['genre'].fillna('unspecified')

for col in ['name', 'artist', 'genre', 'tags']:
    if col in music_df.columns:
        music_df[col] = music_df[col].str.lower().str.strip()

# Step 3: Process User Listening History in Chunks
chunk_size = 100000  # Number of rows per chunk
processed_chunks = []

print("\nProcessing User Listening History in chunks...")
for chunk in pd.read_csv(listening_history_path, chunksize=chunk_size):
    print(f"Processing chunk with {len(chunk)} rows...")
    # Merge the current chunk with the Music Info dataset
    merged_chunk = chunk.merge(music_df, on='track_id', how='inner')
    processed_chunks.append(merged_chunk)

# Step 4: Combine all processed chunks
merged_df = pd.concat(processed_chunks, ignore_index=True)

# Step 5: Save the final merged dataset
merged_df.to_csv(output_path, index=False)
print(f"\nCleaned and merged data saved to {output_path}")

# Step 6: Summary of the merged dataset
print("\nMerged Dataset Info:")
print(merged_df.info())

if 'genre' in merged_df.columns:
    print("\nGenre Distribution in Merged Dataset:")
    print(merged_df['genre'].value_counts())
