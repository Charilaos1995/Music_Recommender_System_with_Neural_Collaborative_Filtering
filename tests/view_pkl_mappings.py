import pickle

# Adjust paths as needed
user_pickle_path = "../NCF_data/user_idx2original.pkl"
song_pickle_path = "../NCF_data/song_idx2original.pkl"

# Load user mapping
with open(user_pickle_path, "rb") as f:
    user_idx2original = pickle.load(f)

# Load song mapping
with open(song_pickle_path, "rb") as f:
    song_idx2original = pickle.load(f)

# Check how many users/songs we have:
print("Number of user mappings:", len(user_idx2original))
print("Number of song mappings:", len(song_idx2original))

# Print a small sample:
print("\n-- First 10 user mappings --")
for idx, original in list(user_idx2original.items())[:10]:
    print(idx, "->", original)

print("\n-- First 10 song mappings --")
for idx, original in list(song_idx2original.items())[:10]:
    print(idx, "->", original)
