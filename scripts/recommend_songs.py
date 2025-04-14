"""
recommend_songs.py

🎧 Interactive Music Recommendation CLI 🎧

This script allows users to interactively generate music recommendations using:
- MLP (neural network based) recommender
- Item-Based Collaborative Filtering recommender

Users can:
- Choose a sample user or select randomly
- Specify how many recommendations to generate
- Optionally save recommendations to a CSV file
"""

# Suppress TensorFlow warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging levels
import logging
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Core libraries
import numpy as np
import pandas as pd
import pickle
import random
from tqdm import tqdm

# Project-specific modules
from src.data.dataset import Dataset
from src.recommenders.MLP import get_model
from src.recommenders.item_cf import ItemCFRecommender


def find_users_in_interaction_range(train_matrix, min_interactions=10, max_interactions=50):
    """
    Finds users who have interacted with a moderate number of songs (not too few, not too many).

    This helps focus the demo on users with meaningful listening history.

    :param train_matrix: Sparse user-item interaction matrix
    :param min_interactions: Minimum number of songs a user must have listened to
    :param max_interactions: Maximum number of songs a user must have listened to
    :return: List of user IDs that fall in the desired interaction range
    """
    return [
        user_id for user_id in range(train_matrix.shape[0])
        if min_interactions <= train_matrix[user_id].count_nonzero() <= max_interactions
    ]

def select_user_from_preview(train_matrix, song_idx2original, df_songs, sample_size=10, max_songs=3):
    """
    Shows a preview of sample users with 10–50 interactions and lets the user choose one.

    :param train_matrix: Sparse matrix of user-item interactions
    :param song_idx2original: Mapping from song indices to original song IDs
    :param df_songs: Metadata for the songs
    :param sample_size: Number of users to preview
    :param max_songs: Number of songs to display per user
    :return: Chosen user ID or None (if user skips)
    """
    print("🎯 Choose a user from the list below:")

    # Find all users who have between 10 and 50 interactions (active but not extreme)
    candidates = find_users_in_interaction_range(train_matrix)

    # Randomly select up to 10 of these users to preview
    sample_users = random.sample(candidates, min(sample_size, len(candidates)))

    # Dictionary to map choice number (1–10) to actual user IDs
    user_map = {}

    # Show the selected users along with a few songs they've listened to
    for i, user_id in enumerate(sample_users, start=1):
        user_map[str(i)] = user_id # Store mapping from selection number to user ID
        print(f"\n{i}. 👤 User {user_id}") # Show user ID

        # Retrieve the binary vector for this user's song interactions
        user_interactions = train_matrix[user_id].toarray().flatten()

        # Get indices of the songs they have interacted with (played)
        listened_indices = np.where(user_interactions > 0)[0][:max_songs] # Limit number of songs displayed

        # Convert internal song indices back to original song IDs
        original_ids = [song_idx2original[idx] for idx in listened_indices]

        # Retrieve song titles and artists for these songs
        user_songs = df_songs[df_songs["song_id"].isin(original_ids)]

        # Print song titles and artists in a readable format
        for _, row in user_songs.iterrows():
            print(f"   🎵 {row['title']} — {row['artist_name']}")

    # Ask the user to select one of the displayed users (or skip)
    print()
    while True:
        choice = input("🔢 Pick a number (1–10) to select a user or press Enter to skip: ").strip()
        if choice == "":
            return None # User skipped selection, will fallback to random later
        elif choice in user_map:
            return user_map[choice] # Return the selected user ID
        else:
            print("❌ Invalid choice. Please try again.")


def print_user_history(user_id, train_matrix, song_idx2original, df_songs, max_to_show=15):
    """
    Display a sample of the songs that a specific user has listened to.

    :param user_id: The numeric ID of the user (internal indexing)
    :param train_matrix: Sparse matrix of user-song interactions
    :param song_idx2original: Mapping from internal song index to original song ID
    :param df_songs: DataFrame containing song metadata (titles, artists)
    :param max_to_show: Maximum number of songs to display for the user
    """

    # Get the interaction row for this user as a dense binary array (1 = listened, 0 = not)
    user_interactions = train_matrix[user_id].toarray().flatten()

    # Find indices of songs that the user has interacted with
    listened_indices = np.where(user_interactions > 0)[0]

    # Display the total number of songs the user has listened to
    print(f"\n👤 User {user_id} has listened to {len(listened_indices)} songs.")

    # If the user has a large history, randomly select a subset to show
    if len(listened_indices) > max_to_show:
        listened_indices = np.random.choice(listened_indices, size=max_to_show, replace=False)

    # Convert internal song indices to original song IDs
    listened_original_ids = [song_idx2original[idx] for idx in listened_indices]

    # Retrieve metadata (title, artist) for the songs the user has listened to
    df_user_songs = df_songs[df_songs["song_id"].isin(listened_original_ids)]

    # Print a formatted list of the songs
    for _, row in df_user_songs.iterrows():
        print(f"  🎵 ID: {row['song_id']}, Title: {row['title']}, Artist: {row['artist_name']}")


def recommend_for_user_MLP(model, user_id, train_matrix, num_songs, top_k=10, batch_size=1000):
    """
    Generate top-K music recommendations for a user using the trained MLP model.

    :param model: Trained MLP model for collaborative filtering
    :param user_id: Internal numeric user ID
    :param train_matrix: Sparse matrix of user-song interactions (user-item)
    :param num_songs: Total number of songs in the dataset
    :param top_k: Number of top recommendations to return
    :param batch_size: Number of samples per prediction batch
    :return: List of internal song indices recommended to the user
    """

    # Convert user row to dense format and find which songs the user has listened to
    user_interactions = train_matrix[user_id].toarray().flatten()
    listened = set(np.where(user_interactions > 0)[0]) # Indices of songs the user has interacted with

    # Filter out the songs the user already knows — we only want to recommend unseen ones
    candidate_songs = np.array([i for i in range(num_songs) if i not in listened])

    # If the user has seen everything (unlikely), we return nothing
    if candidate_songs.size == 0:
        return []

    # Prepare the model input: replicate the user ID to match each candidate song
    user_inputs = np.full_like(candidate_songs, fill_value=user_id)

    # Log progress to the terminal using tqdm
    tqdm.write("🔄 Predicting with MLP model...")

    # Predict the relevance score for each (user, song) pair using the model
    predictions = model.predict([user_inputs, candidate_songs], batch_size=batch_size, verbose=0)

    # Sort candidates by their predicted score in descending order and select the top-K
    top_indices = np.argsort(predictions.flatten())[::-1][:top_k]

    # Return the top-K recommended song indices
    return candidate_songs[top_indices]


def recommend_for_user_itemcf(recommender, user_id, num_songs, top_k=10):
    """
    Generate top-K music recommendations for a user using Item-based Collaborative Filtering (ItemCF).

    :param recommender: Instance of ItemCFRecommender (already fitted with item similarity matrix)
    :param user_id: Internal numeric user ID
    :param num_songs: Total number of songs in the dataset
    :param top_k: Number of top recommendations to return
    :return: Numpy array of internal song IDs recommended to the user
    """

    # Get the user's interaction history as a dense array (1 if interacted, 0 otherwise)
    user_interactions = recommender.train_matrix[user_id].toarray().flatten()

    # Find songs the user has already listened to
    listened = set(np.where(user_interactions > 0)[0])

    # Get a list of candidate songs the user has not listened to
    candidate_songs = [i for i in range(num_songs) if i not in listened]

    # Get the actual items (song indices) the user has interacted with
    interacted_items = np.where(user_interactions > 0)[0]

    # Inform the user that scoring has started
    tqdm.write("🔄 Computing item similarity scores...")

    # For each candidate song, compute a relevance score based on similarity with songs the user has liked
    scores = [
        np.sum(recommender.item_similarity[song_id, interacted_items]) if interacted_items.size > 0 else 0.0
        for song_id in candidate_songs
    ]

    # Get indices of top-K highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Return the top-K candidate song IDs
    return np.array(candidate_songs)[top_indices]


def save_recommendations_csv(filename, user_id, df_mlp, df_itemcf):
    """
    Save generated song recommendations to a CSV file.

    The output file includes song metadata for each recommendation, along with the user ID and model name.
    This makes it easy to inspect or share recommendations from both models in a clean, readable format.

    :param filename: Desired name (or path) of the CSV file
    :param user_id: Internal numeric user ID the recommendations were generated for
    :param df_mlp: DataFrame of songs recommended by the MLP model (or None if skipped)
    :param df_itemcf: DataFrame of songs recommended by the ItemCF model (or None if skipped)
    """
    output = []

    # Collect MLP recommendations, if available
    if df_mlp is not None:
        for _, row in df_mlp.iterrows():
            output.append({
                "user_id": user_id,
                "model": "MLP",
                "song_id": row['song_id'],
                "title": row['title'],
                "artist": row['artist_name']
            })

    # Collect ItemCF recommendations, if available
    if df_itemcf is not None:
        for _, row in df_itemcf.iterrows():
            output.append({
                "user_id": user_id,
                "model": "ItemCF",
                "song_id": row['song_id'],
                "title": row['title'],
                "artist": row['artist_name']
            })

    # If the user provided a plain filename (no folder), store it in 'recommendations/'
    if not os.path.dirname(filename):
        os.makedirs("recommendations", exist_ok=True)
        filename = os.path.join("recommendations", filename)

    # Save to disk as a clean CSV
    pd.DataFrame(output).to_csv(filename, index=False)

    print(f"\n✅ Recommendations saved to: {filename}")


def run_recommendation_pipeline(top_k=10, history_limit=15, model_choice="both", user_id=None, save_csv=None):
    """
    Runs the full recommendation pipeline for a selected user using either MLP, ItemCF, or both.

    This function loads the necessary models and data, selects a user, prints their listening history,
    generates Top-K song recommendations using the selected model(s), displays the results in the terminal,
    and optionally saves the results to a CSV file.

    :param top_k: Number of song recommendations to return
    :param history_limit: Maximum number of songs from the user's history to display
    :param model_choice: 'mlp', 'itemcf', or 'both' to select which model(s) to use
    :param user_id: Optional internal user ID. If None, a random user will be selected
    :param save_csv: Optional filename for saving the recommendations to a CSV
    """

    # Load dataset and retrieve train matrix
    dataset = Dataset("NCF_data/")
    train_matrix = dataset.trainMatrix
    num_users, num_songs = train_matrix.shape
    print(f"\n📊 Dataset: {num_users} users, {num_songs} songs")

    # Load the trained MLP model if needed
    mlp_model = None
    if model_choice in ("mlp", "both"):
        print("🔍 Loading MLP model...")
        mlp_model = get_model(num_users, num_songs, [64, 32, 16, 8], [0, 0, 0, 0])
        mlp_model.load_weights("models/MLP_[64,32,16,8].h5")

    # Initialize the Item-based Collaborative Filtering recommender if selected
    item_cf = ItemCFRecommender(train_matrix) if model_choice in ("itemcf", "both") else None

    # Load the song metadata
    df_songs = pd.read_csv("NCF_data/cleaned_merged_music_data.csv").drop_duplicates("song_id").reset_index(drop=True)

    # Load the song index-to-original ID mapping
    with open("NCF_data/song_idx2original.pkl", "rb") as f:
        song_idx2original = pickle.load(f)

    # Select a user
    if user_id is None:
        # If no user is given, randomly pick one with a moderate number of interactions
        candidates = find_users_in_interaction_range(train_matrix)
        if not candidates:
            print("❌ No users in the interaction range.")
            return
        user_id = random.choice(candidates)
        print(f"🎯 Randomly selected user: {user_id}")
    else:
        print(f"🎯 Using selected user: {user_id}")

    # Print this user's listening history
    print_user_history(user_id, train_matrix, song_idx2original, df_songs, max_to_show=history_limit)

    # Containers to hold recommendations
    df_mlp = df_itemcf = None

    # Generate MLP-based recommendations
    if mlp_model:
        print("\n=== MLP Recommendations ===")
        mlp_reco = recommend_for_user_MLP(mlp_model, user_id, train_matrix, num_songs, top_k)
        mlp_original = [song_idx2original[idx] for idx in mlp_reco]
        df_mlp = df_songs[df_songs["song_id"].isin(mlp_original)]

        # Display recommendations
        for _, row in df_mlp.iterrows():
            print(f"  ⭐ ID: {row['song_id']}, Title: {row['title']}, Artist: {row['artist_name']}")

    # Generate ItemCF-based recommendations
    if item_cf:
        print("\n=== ItemCF Recommendations ===")
        itemcf_reco = recommend_for_user_itemcf(item_cf, user_id, num_songs, top_k)
        itemcf_original = [song_idx2original[idx] for idx in itemcf_reco]
        df_itemcf = df_songs[df_songs["song_id"].isin(itemcf_original)]

        # Display recommendations
        for _, row in df_itemcf.iterrows():
            print(f"  ⭐ ID: {row['song_id']}, Title: {row['title']}, Artist: {row['artist_name']}")

    # Save recommendations to CSV, if requested
    if save_csv:
        save_recommendations_csv(save_csv, user_id, df_mlp, df_itemcf)


def main():
    """
    Interactive CLI for generating music recommendations using MLP, ItemCF, or both.
    Repeats the process until the user chooses to exit.
    """
    print("🎶 Welcome to the Interactive Music Recommender 🎶\n")

    try:
        while True:
            # Load the dataset
            dataset = Dataset("NCF_data/")
            train_matrix = dataset.trainMatrix
            df_songs = pd.read_csv("NCF_data/cleaned_merged_music_data.csv").drop_duplicates("song_id").reset_index(drop=True)
            with open("NCF_data/song_idx2original.pkl", "rb") as f:
                song_idx2original = pickle.load(f)

            # Let user pick a sample user
            user_id = select_user_from_preview(train_matrix, song_idx2original, df_songs)

            # Prompt for Top-K
            while True:
                try:
                    top_k = int(input("🔢 How many songs should we recommend (Top-K)? [default=10]: ") or 10)
                    break
                except ValueError:
                    print("❌ Please enter a valid integer.")

            # Prompt for history length
            while True:
                try:
                    history_limit = int(input("🧠 How many songs from user history to display? [default=15]: ") or 15)
                    break
                except ValueError:
                    print("❌ Please enter a valid integer.")

            # Prompt for model selection
            model_choice = ""
            while model_choice not in ("mlp", "itemcf", "both"):
                model_choice = input("🤖 Choose model (mlp / itemcf / both) [default=both]: ").strip().lower() or "both"
                if model_choice not in ("mlp", "itemcf", "both"):
                    print("❌ Invalid choice. Please enter 'mlp', 'itemcf', or 'both'.")

            # Prompt to save CSV
            save_csv = input("💾 Save recommendations to CSV? Enter filename or leave blank to skip: ").strip() or None

            # Run the recommender pipeline
            run_recommendation_pipeline(
                top_k=top_k,
                history_limit=history_limit,
                model_choice=model_choice,
                user_id=user_id,
                save_csv=save_csv
            )

            # Ask if user wants to run again
            again = input("\n🔁 Run another recommendation? (y/n): ").strip().lower()
            if again != "y":
                print("\n👋 Thanks for using the music recommender!")
                break

    except Exception as e:
        print(f"\n❌ Something went wrong: {e}")


if __name__ == "__main__":
    main()
