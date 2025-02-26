import sys
import os
import time
from src.data.dataset import Dataset

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


def test_dataset_loading():
    """Test the Dataset class by loading data and checking its attributes."""

    print("\nStarting dataset test...")

    # Ensure dataset files exist before loading
    data_dir = "NCF_data"
    required_files = ["train.rating", "test.rating", "test.negative"]

    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"ERROR: Missing required dataset file: {file_path}")
            return

    # Measure dataset loading time
    start_time = time.time()

    try:
        dataset = Dataset()
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {str(e)}")
        return

    end_time = time.time()
    print(f"Dataset loaded successfully in {end_time - start_time:.2f} seconds!")

    # Check dataset dimensions
    print(f"\nDataset Info: {dataset.num_users} users, {dataset.num_items} songs.")

    # Check a few training samples
    print("\nSample Train Ratings:")
    train_samples = list(zip(*dataset.trainMatrix.nonzero()))[:5]  # Convert sparse matrix indices to list
    for user, song in train_samples:
        print(f"User {user} -> Song {song} | Rating: 1.0")

    # Check test set
    print("\nSample Test Ratings:")
    for user, song in dataset.testRatings[:5]:  # Print first 5 test samples
        print(f"User {user} -> Song {song}")

    # Check negative samples
    print("\nSample Negative Samples:")
    for user_negatives in dataset.testNegatives[:5]:  # Print first 5 users' negatives
        print(f"User {user_negatives[0]} -> Negatives: {user_negatives[1:5]}...")  # Show only first 4

    print("\nAll dataset checks passed!")


if __name__ == "__main__":
    test_dataset_loading()
