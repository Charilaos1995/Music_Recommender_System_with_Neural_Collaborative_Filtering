import pandas as pd
import os

# Define data directory
data_dir = "../data"

# Function to preview file contents
def preview_file(file_path, num_lines=5):
    print(f"\nPreview of {file_path}:")
    with open(file_path, "r") as f:
        for _ in range(num_lines):
            print(f.readline().strip())

# Check train.rating
train_file = os.path.join(data_dir, "train.rating")
print("\nChecking train.rating...")
df_train = pd.read_csv(train_file, sep="\t", header=None, names=["user_id", "song_id", "rating", "timestamp"])
print(df_train.info())  # Check data types
preview_file(train_file)

# Check test.rating
test_file = os.path.join(data_dir, "test.rating")
print("\nChecking test.rating...")
df_test = pd.read_csv(test_file, sep="\t", header=None, names=["user_id", "song_id"])
print(df_test.info())  # Check data types
preview_file(test_file)

# Check test.negative
test_negative_file = os.path.join(data_dir, "test.negative")
print("\nChecking test.negative...")
preview_file(test_negative_file)

# Validate negative samples per user
with open(test_negative_file, "r") as f:
    first_line = f.readline().strip().split("\t")
    print(f"\nFirst user in test.negative has {len(first_line) - 1} negative samples (Expected: 99)")
