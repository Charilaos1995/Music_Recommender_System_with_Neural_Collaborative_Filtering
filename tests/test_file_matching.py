import os

data_dir = "../data"

# File paths
test_file = os.path.join(data_dir, "test.rating")
negative_file = os.path.join(data_dir, "test.negative")

# Count the number of lines
num_test_ratings = sum(1 for _ in open(test_file, "r"))
num_negative_samples = sum(1 for _ in open(negative_file, "r"))

print(f"Test.rating count: {num_test_ratings}")
print(f"Test.negative count: {num_negative_samples}")

# Check if lengths match
if num_test_ratings == num_negative_samples:
    print("The files match correctly.")
else:
    print("Mismatch detected! test.negative does not align with test.rating.")
