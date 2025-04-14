"""
Dataset processing module for the Neural Collaborative Filtering (NCF) implementation
of the Music Recommender System.

This script handles loading and formatting of:
- Train ratings (user-item interactions)
- Test ratings (ground truth samples)
- Negative samples (used for ranking evaluation)

All datasets are converted into formats suitable for model input and evaluation.
"""

import os
import scipy.sparse as sp # For creating a sparse matrix of user-item interactions
import numpy as np
import logging

# Configure logging only if it's not already set up externally (e.g., from CLI or other modules)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

class Dataset:
    def __init__(self, data_dir=None):
        """
        Initializes the Dataset object by loading rating and negative sample files,
        and generating matrix and list structures required for NCF training and evaluation.

        :param data_dir: (str or None) The directory path where the data files (train.rating, test.rating, test.negative) are stored.
                                       If None, it defaults to the path: ../../NCF_data relative to this file's location.
        """
        # Use custom directory if provided; else default to ../../NCF_data relative to this script
        self.data_dir = data_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../NCF_data"))

        # Construct full file paths for train, test, and negative data
        self.train_file = os.path.join(self.data_dir, "train.rating")
        self.test_file = os.path.join(self.data_dir, "test.rating")
        self.negative_file = os.path.join(self.data_dir, "test.negative")

        # Check if all required files exist, else raise an error
        self._check_if_files_exist()

        # Load training ratings into a sparse matrix
        self.trainMatrix = self.load_rating_file_as_matrix(self.train_file)

        # Load test ratings into a list of (user, item) tuples
        self.testRatings = self.load_rating_file_as_list(self.test_file)

        # Load negative samples into a list of [item1, item2, ..., item99] per user
        self.testNegatives = self.load_negative_file(self.negative_file)

        # Ensure testRatings and testNegatives have the same number of entries
        assert len(self.testRatings) == len(self.testNegatives), "Mismatch between test ratings and negatives."

        # Extract total number of users and items based on shape of training matrix
        self.num_users, self.num_items = self.trainMatrix.shape

        # Log basic dataset summary
        logging.info(
            f"Dataset loaded: {self.num_users} users, "
            f"{self.num_items} songs, "
            f"{len(self.trainMatrix.nonzero()[0])} interactions"
        )

    def _check_if_files_exist(self):
        """
        Checks for the existence of the required dataset files in the specified directory.

        This includes:
        - train.rating: The training data containing user-item interactions
        - test.rating: The test set for evaluation
        - test.negative: The negative samples for ranking evaluation

        :raises: (FileNotFoundError) If any of the required files are missing in the dataset directory.
        """
        # Loop through each required file and verify its existence
        for file in [self.train_file, self.test_file, self.negative_file]:
            if not os.path.exists(file):
                # Raise an error if a file is missing
                raise FileNotFoundError(f"Missing required file: {file}")

    def load_rating_file_as_list(self, filename):
        """
        Loads the test.rating file and parses it into a list of [user_id, song_id] pairs.

        :param filename: (str) the path to the .rating file containing the tab-seperated user and song ID's.
        :return: list of list [int, int] - A list where each entry is a [user_id, song_id] interaction for evaluation.
        """
        rating_list = []
        # Open the file and read it line by line
        with open(filename, "r") as f:
            # For each line in the file
            for line in f:
                # Strip newline characters and split each line into components
                arr = line.strip().split("\t")
                # Convert user and song IDs to integers
                user, song = int(arr[0]), int(arr[1])
                # Append the interaction pair to the list
                rating_list.append([user, song])

        return rating_list

    def load_negative_file(self, filename):
        """
        Loads a file containing negative song samples for each user (e.g., test.negative).

        Each line in the file contains a user ID followed by a list of song IDs the user has not interacted with.
        These are used during evaluation for ranking among negative items.

        :param filename: (str) Path to the test.negative file with tab-separated values.

        :return: list of lists [int] -  A list where each element is a list of negative song IDs (as integers) for a specific user.
                                        The first value (user ID) is ignored in the result because the ordering aligns with test.rating.
        """
        negative_list = []

        # Open the file and read it line by line
        with open(filename, "r") as f:
            for line in f:
                # Remove any trailing whitespace and split the line by tab
                arr = line.strip().split("\t")
                # Ignore the first element (user ID), convert the rest to integers
                negatives = list(map(int, arr[1:]))
                # Append the list of negative song IDs to the overall list
                negative_list.append(negatives)

        return negative_list

    def load_rating_file_as_matrix(self, filename):
        """
        Loads a rating file and converts it into a sparse matrix of shape (num_users, num_items),
        where each entry (u, i) in the matrix is set to 1 if user u has interacted with item i.
        This is suitable for implicit feedback datasets like music listening history.

        :param filename: Path to the 'train.rating' file, containing tab-separated lines:
                         user_id \t song_id \t rating
        :return: A scipy sparse CSR matrix (Compressed Sparse Row) where rows represent users,
                 columns represent songs, and values are 1.0 if interaction exists
        """
        num_users, num_items = 0, 0 # Track max user and song IDs
        interactions = [] # Store all (user, song) pairs

        # First pass: determine matrix dimensions & collect interactions
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split("\t")
                u, i = int(arr[0]), int(arr[1]) # user_id and song_id
                num_users = max(num_users, u)   # Update max user ID
                num_items = max(num_items, i)   # Update max song ID
                interactions.append((u, i))     # Save the interaction pair

        # Separate user and song IDs into row and column indices
        row, col = zip(*interactions)  # E.g., row = (0, 1, 2), col = (5, 3, 8)

        # Create a NumPy array with ones (for implicit feedback)
        data = np.ones(len(row), dtype=np.float32)

        # Create a sparse matrix of shape (num_users+1, num_items+1) with 1.0 values at (user, song) positions
        mat = sp.csr_matrix((data, (row, col)), shape=(num_users + 1, num_items + 1), dtype=np.float32)

        logging.info(f"Loaded {len(interactions)} interactions into sparse matrix.")

        return mat

    def get_train_instances(self, num_negatives):
        """
        Generates training instances for the NCF model by combining real (positive) interactions
        with artificially created (negative) interactions using negative sampling.

        For every positive interaction (user, song), this function generates `num_negatives`
        negative samples, where the song is randomly selected such that the user has *not*
        interacted with it.

        :param num_negatives: (int) Number of negative samples to generate for every positive instance
        :return: Three lists — user_input, song_input, and labels — to be fed into the model for training.
                Each index corresponds to one training instance:
                - user_input[i] = user ID
                - song_input[i] = song ID
                - labels[i] = 1 (positive) or 0 (negative)
        """
        user_input, song_input, labels = [], [], []

        # Get total number of users and songs from the shape of the interaction matrix
        num_users, num_songs = self.trainMatrix.shape

        # Loop through each (user, song) pair in the training matrix that has a positive interaction
        for (u, s) in zip(*self.trainMatrix.nonzero()):
            # Positive instance
            user_input.append(u)
            song_input.append(s)
            labels.append(1)

            # Negative sampling
            for _ in range(num_negatives):
                # Randomly select a song ID
                neg_s = np.random.randint(num_songs)

                # Keep generating new song IDs until we find one the user hasn't interacted with
                while self.trainMatrix[u, neg_s] != 0: # Ensure it's not an actual interaction
                    neg_s = np.random.randint(num_songs)

                # Add the negative sample (same user, different song)
                user_input.append(u)
                song_input.append(neg_s)
                labels.append(0)

        return user_input, song_input, labels
