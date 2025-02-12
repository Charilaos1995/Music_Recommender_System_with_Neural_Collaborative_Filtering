import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.data.dataset import Dataset

def test_dataset_loading():
    """Test the Dataset class by loading data and checking its attributes."""

    print("\nLoading dataset...")
    dataset = Dataset()

    # Check dataset dimensions
    print(f"Dataset Loaded: {dataset.num_users} users, {dataset.num_items} songs.")

    # Check a few training samples
    print("\nSample Train Ratings:")
    train_samples = list(dataset.trainMatrix.items())[:5] # Get the first 5 interactions
    for (user, song), value in train_samples:
        print(f"User {user} -> Song {song} | Rating: {value}")

    # Check test set
    print("\nSample Test Ratings:")
    for user, song in dataset.testRatings[:5]: # Print first 5 test samples
        print(f"User {user} -> Song {song}")

    # Check negative samples
    print("\nSample Negative Samples:")
    for user_negatives in dataset.testNegatives[:5]: # Print the first 5 users' negatives
        print(f"User {user_negatives[0]} -> Negatives: {user_negatives[1:5]}...") # Show only the first 4

    print("\nAll dataset checks passed!")

if __name__ == "__main__":
    test_dataset_loading()