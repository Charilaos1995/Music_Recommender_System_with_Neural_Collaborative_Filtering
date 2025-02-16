import pandas as pd
import numpy as np
import os
import random

# Data directory to save the generated files
data_dir = "../NCF_data"

# Load the dataset
file_path = os.path.join(data_dir, "cleaned_merged_music_data.csv")
df = pd.read_csv(file_path)

# Select the relevant columns
df = df[['user_id', 'song_id', 'listen_count']]

# Convert 'user_id' and 'song_id' to categorical integer indices
df['user_id'] = df['user_id'].astype('category').cat.codes
df['song_id'] = df['song_id'].astype('category').cat.codes

# Assign implicit feedback (rating = 1 for listened songs)
df['rating'] = 1

# Generate timestamp
df['timestamp'] = np.random.randint(1500000000, 1600000000, df.shape[0])

# Ensure the output directory exists
os.makedirs(data_dir, exist_ok=True)

# Save the train.rating file
train_file = os.path.join(data_dir, "train.rating")
df[['user_id', 'song_id', 'rating', 'timestamp']].to_csv(train_file, sep="\t", index=False, header=False)
print(f"Saved: {train_file}")

# Generate test.rating (one song per user)
test_ratings = df.groupby('user_id').sample(n=1, random_state=42)[['user_id', 'song_id']]
test_file = os.path.join(data_dir, "test.rating")
test_ratings.to_csv(test_file, sep="\t", index=False, header=False)
print(f"Saved: {test_file}")

# Generate test.negative (sample 99 negative items per user)
all_songs_ids = set(df['song_id'].unique())
negative_samples = []

for user, song in test_ratings.itertuples(index=False):
    user_songs = set(df[df['user_id'] == user]['song_id'])
    negatives = list(all_songs_ids - user_songs)  # Songs the user hasn't listened to

    if len(negatives) < 99:
        print(f"User {user} has only {len(negatives)} possible negative samples. Filling with all available.")
        sampled_negatives = negatives  # Use all available negatives
    else:
        sampled_negatives = random.sample(negatives, 99)  # Sample 99 negatives

    negative_samples.append([user, *sampled_negatives])

print(f"Generated {len(negative_samples)} negative samples (expected: {len(test_ratings)})")


# Save test.negative file
test_negative_file = os.path.join(data_dir, "test.negative")

# Convert list to DataFrame
df_negatives = pd.DataFrame(negative_samples)

# Save the file
df_negatives.to_csv(test_negative_file, sep="\t", index=False, header=False)

print(f"Successfully saved test.negative with {df_negatives.shape[0]} rows.")