"""
prepare_ncf_data.py

This script prepares interaction data for use with Neural Collaborative Filtering (NCF) models.

Functionality:
--------------
- Loads merged user-song interaction data from the cleaned dataset.
- Converts original user and song IDs to numeric indices (required for model training).
- Saves mapping files (index → original ID) as Pickle files for later use.
- Assigns implicit feedback scores (rating=1) to all observed interactions.
- Generates three data files required by NCF-based recommenders:
    1. train.rating      → All (user, song, rating) interactions
    2. test.rating       → One positive interaction per user (for evaluation)
    3. test.negative     → 99 negative samples per user (for ranking metrics like HR@K)

Outputs:
--------
- train.rating         (tab-separated, no headers)
- test.rating          (tab-separated, no headers)
- test.negative        (tab-separated, no headers)
- user_idx2original.pkl and song_idx2original.pkl for mapping back to original IDs

This script is run once, prior to training or evaluating any models.
"""

import os
import pandas as pd
import random
import pickle

# Define the directory containing the source and output files
data_dir = "../NCF_data"

# Load the cleaned and merged dataset
file_path = os.path.join(data_dir, "cleaned_merged_music_data.csv")
df = pd.read_csv(file_path)

# Keep only the relevant columns for recommendation tasks
df = df[['user_id', 'song_id', 'listen_count']]

# Convert user and song IDs to categorical values so we can map them to unique integers
df['user_id'] = df['user_id'].astype('category')
user_categories = df['user_id'].cat.categories # Save the original string IDs

df['song_id'] = df['song_id'].astype('category')
song_categories = df['song_id'].cat.categories # Save the original string IDs

# Create mappings: integer index -> original string ID
user_idx2original = {i: cat for i, cat in enumerate(user_categories)}
song_idx2original = {i: cat for i, cat in enumerate(song_categories)}

# Convert to numeric codes (stating from 0)
df['user_id'] = df['user_id'].cat.codes
df['song_id'] = df['song_id'].cat.codes

# Assign implicit feedback (binary rating — 1 for every interaction)
df['rating'] = 1

# Save the index-to-original mappings for future use
# At this point, we've converted all user and song IDs to numeric format (0, 1, 2, ...),
# which is required for training models like MLP (uses embedding layers that expect integer indices).
# However, later when we want to display results (like showing actual song titles or usernames),
# we'll need to reverse that mapping from numeric index to original string (user_id, song_id).
# So we store these mappings in .pkl (Pickle) files.

# Ensure the output directory exists
os.makedirs(data_dir, exist_ok=True)

# Save the user index-to-original ID mapping as a pickle file
# Format: {0: "user_hash_1", 1: "user_hash_2", ...}
with open(os.path.join(data_dir, "user_idx2original.pkl"), "wb") as f:
    pickle.dump(user_idx2original, f)

# Save the song index-to-original ID mapping similarly
# Format: {0: "song_id_1", 1: "song_id_2", ...}
with open(os.path.join(data_dir, "song_idx2original.pkl"), "wb") as f:
    pickle.dump(song_idx2original, f)

# Save the training dataset (format: user_id, song_id, rating) — required by NCF
train_file = os.path.join(data_dir, "train.rating")
df[['user_id', 'song_id', 'rating']].to_csv(train_file, sep="\t", index=False, header=False)
print(f"Saved: {train_file}")

# Generate test.rating: one random positive sample per user
test_ratings = df.groupby('user_id').sample(n=1, random_state=42)[['user_id', 'song_id']]
test_file = os.path.join(data_dir, "test.rating")
test_ratings.to_csv(test_file, sep="\t", index=False, header=False)
print(f"Saved: {test_file}")

# Generate test.negative: for each test rating, generate up to 99 negative samples (not interacted)
all_songs_ids = set(df['song_id'].unique())
negative_samples = []

# For every (user, song) pair in the test set (test.rating)
# test_ratings contains one positive interaction per user, which we'll use to create negatives for evaluation.
for user, song in test_ratings.itertuples(index=False):
    # Get the set of songs the user has already interacted with (positive interactions)
    # This helps us avoid recommending songs they’ve already seen.
    user_songs = set(df[df['user_id'] == user]['song_id'])

    # Compute the negative song candidates for this user
    # We subtract the songs they've seen from the full song set to get songs they've never interacted with
    negatives = list(all_songs_ids - user_songs)

    # Randomly sample up to 99 negative song IDs from the remaining ones
    # These will simulate "songs not liked or clicked", which we use to test how well the model ranks the true positive
    sampled_negatives = random.sample(negatives, min(99, len(negatives))) # Sample up to 99 negatives

    # Create a list: [user_id, neg_song_1, neg_song_2, ..., neg_song_99]
    # This format is expected by the evaluation function in the NCF framework (test.negative file)
    negative_samples.append([user, *sampled_negatives])

print(f"Generated {len(negative_samples)} negative samples (expected: {len(test_ratings)})")

# Save test.negative file
test_negative_file = os.path.join(data_dir, "test.negative")

# Convert list to DataFrame
df_negatives = pd.DataFrame(negative_samples)

# Save the file
df_negatives.to_csv(test_negative_file, sep="\t", index=False, header=False)

print(f"Successfully saved test.negative with {df_negatives.shape[0]} rows.")