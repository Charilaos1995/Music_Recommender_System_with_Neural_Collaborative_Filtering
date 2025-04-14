"""
verify_generated_files.py

This script is used to verify the integrity and format of the interaction files
prepared for training and evaluating Neural Collaborative Filtering (NCF) models.

It performs the following checks:
- Loads and previews the contents of:
    - train.rating      (contains user-song-rating-timestamp data)
    - test.rating       (one song per user for evaluation)
    - test.negative     (list of 99 negative songs per user)
- Confirms the structure and column data types of each file.
- Validates that test.negative contains the expected number of negative items per user.

Intended as a quick debugging tool before training or evaluation begins.
"""

import pandas as pd
import os

# Define data directory where the generated NCF files are stored
data_dir = "../NCF_data"

def preview_file(file_path, num_lines=5):
    """
    Displays the first few lines of a text file (e.g., a dataset or rating file).

    This utility function is helpful to quickly verify the structure and sample contents
    of a file such as train.rating, test.rating, or test.negative without loading the entire dataset.

    :param file_path: (str) The full or relative path to the file to preview.
    :param num_lines: (int) optional (default=5) - The number of lines from the beginning of the file to display.
    :return: None
    """
    print(f"\nPreview of {file_path}:")
    # Open the file for reading
    with open(file_path, "r") as f:
        # Loop through the first `num_lines` lines in the file
        for _ in range(num_lines):
            # Read and print each line after stripping any trailing newline characters
            print(f.readline().strip())

# Check train.rating file
train_file = os.path.join(data_dir, "train.rating")
print("\nChecking train.rating...")
df_train = pd.read_csv(train_file, sep="\t", header=None, names=["user_id", "song_id", "rating", "timestamp"])
print(df_train.info())  # Display structure and types of columns
preview_file(train_file) # Show first few lines

# Check test.rating
test_file = os.path.join(data_dir, "test.rating")
print("\nChecking test.rating...")
df_test = pd.read_csv(test_file, sep="\t", header=None, names=["user_id", "song_id"])
print(df_test.info())  # Check data types
preview_file(test_file)

# Check test.negative file
test_negative_file = os.path.join(data_dir, "test.negative")
print("\nChecking test.negative...")
preview_file(test_negative_file)

# Validate the number of negative samples for the first user
with open(test_negative_file, "r") as f:
    first_line = f.readline().strip().split("\t")
    print(f"\nFirst user in test.negative has {len(first_line) - 1} negative samples (Expected: 99)")
