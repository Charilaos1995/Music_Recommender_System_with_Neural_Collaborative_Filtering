import pandas as pd

# File paths
users_path = '../NCF_data/triplets_file.csv'
songs_path = '../NCF_data/song_data.csv'
output_path = '../NCF_data/cleaned_merged_music_data.csv'
sampled_output_path = '../NCF_data/sampled_music_data.csv'

# Load the Datasets
users_df = pd.read_csv(users_path)
songs_df = pd.read_csv(songs_path)

# Check for missing values in the datasets
print("Missing values in users dataset:")
print(users_df.isnull().sum())
print("Missing values in songs dataset:")
print(songs_df.isnull().sum())

# Merge the datasets on song_id
music_df = pd.merge(users_df, songs_df, on='song_id', how='left')

# Drop duplicate rows if any
music_df = music_df.drop_duplicates()

# Take a sample of 10,000 rows for quick results
music_sample_df = music_df.sample(n=10000, random_state=42)

# Save both the full dataset and the sampled dataset
music_df.to_csv(output_path, index=False)
music_sample_df.to_csv(sampled_output_path, index=False)

print(f"Full dataset saved to: {output_path}")
print("Sampled dataset saved to: NCF_data/sampled_music_data.csv")

# Display a summary of the sampled dataset
print('\nSampled DataFrame:')
print(music_sample_df.head())

# Display grouped statistics for debugging
music_grouped = music_sample_df.groupby(['title', 'artist_name']).agg({'listen_count':'sum'}).reset_index()
print("\nTop Songs by Listen Count:")
print(music_grouped.sort_values(['listen_count', 'title'], ascending=[0,1]).head())