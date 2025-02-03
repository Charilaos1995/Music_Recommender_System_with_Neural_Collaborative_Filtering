import pandas as pd

# File paths
datasets = {
    "cleaned_merged_music_data": "../data/cleaned_merged_music_data.csv",
    "sampled_music_data": "../data/sampled_music_data.csv"
}

# Function to analyze each dataset
def analyze_dataset(file_path):
    df = pd.read_csv(file_path)

    # Get unique counts
    unique_users = df['user_id'].nunique()
    unique_songs = df['song_id'].nunique()

    # Count interactions (number of songs per user)
    interactions_per_user = df.groupby('user_id')['song_id'].count()

    # Stats summary
    print(f"\nDataset: {file_path}")
    print(f"Unique Users: {unique_users}")
    print(f"Unique Songs: {unique_songs}")
    print(f"Total Interactions: {interactions_per_user.sum()}")
    print(f"Average songs per user: {interactions_per_user.mean():.2f}")
    print(f"Max songs by a single user: {interactions_per_user.max()}")

# Run the analysis for each dataset
for name, path in datasets.items():
    analyze_dataset(path)