"""
dataset_analysis.py

Quickly analyzes the user-song interaction datasets for:
- Unique user and song counts
- Total interactions
- User interaction statistics (mean and max songs per user)

Inputs:
    ../NCF_data/cleaned_merged_music_data.csv : Full merged dataset
    ../NCF_data/sampled_music_data.csv        : Random sample of 10,000 rows

Outputs:
    Printed dataset statistics summary to terminal
"""

import pandas as pd

# Dataset paths
datasets = {
    "cleaned_merged_music_data": "../NCF_data/cleaned_merged_music_data.csv",
    "sampled_music_data": "../NCF_data/sampled_music_data.csv"
}

def analyze_dataset(file_path, name):
    """
    Analyzes a single dataset and prints summary stats.

    :param file_path: (str) Path to the CSV file
    :param name: (str) Display name of the dataset
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Count how many unique users are present in the dataset
    unique_users = df['user_id'].nunique()

    # Count how many unique songs appear in the dataset
    unique_songs = df['song_id'].nunique()

    # Group the dataset by user_id and count how many songs each user has interacted with
    # This gives us the number of interactions per user
    interactions_per_user = df.groupby('user_id')['song_id'].count()

    # Print a formatted summary of the dataset
    print(f"\nDataset: {file_path}")
    print(f"Unique Users: {unique_users}")
    print(f"Unique Songs: {unique_songs}")
    print(f"Total Interactions: {interactions_per_user.sum()}")
    print(f"Average songs per user: {interactions_per_user.mean():.2f}")
    print(f"Max songs by a single user: {interactions_per_user.max()}")

# Run the analysis for each dataset
if __name__ == "__main__":
    # Loop through each dataset defined in the dictionary
    for name, path in datasets.items():
        # Analyze the dataset and print results
        analyze_dataset(path, name)