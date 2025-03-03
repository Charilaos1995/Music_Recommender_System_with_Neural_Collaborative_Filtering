"""
Dataset processing for the Neural Collaborative Filtering implementation of the Music Recommender System
Handles loading of train, test, and negative samples for NCF.
"""
import os
import scipy.sparse as sp
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Dataset:
    def __init__(self, data_dir=None):
        """
        Initializes the dataset, loading train, test, and negative samples.
        :param data_dir:
        """
        # Set default data directory if not provided
        self.data_dir = data_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../NCF_data"))

        # File paths
        self.train_file = os.path.join(self.data_dir, "train.rating")
        self.test_file = os.path.join(self.data_dir, "test.rating")
        self.negative_file = os.path.join(self.data_dir, "test.negative")

        # Check if the files exist before loading them
        self._check_if_files_exist()

        # Load the datasets
        self.trainMatrix = self.load_rating_file_as_matrix(self.train_file)
        self.testRatings = self.load_rating_file_as_list(self.test_file)
        self.testNegatives = self.load_negative_file(self.negative_file)

        assert len(self.testRatings) == len(self.testNegatives), "Mismatch between test ratings and negatives."

        # Set dataset dimensions
        self.num_users, self.num_items = self.trainMatrix.shape
        logging.info(f"Dataset loaded: {self.num_users} users, {self.num_items} songs, {len(self.trainMatrix.nonzero()[0])} interactions")

    def _check_if_files_exist(self):
        """Check if the required dataset files exist."""
        for file in [self.train_file, self.test_file, self.negative_file]:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Missing required file: {file}")

    def load_rating_file_as_list(self, filename):
        """Load the test rating file as a list of (user, song) pairs."""
        rating_list = []
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split("\t")
                user, song = int(arr[0]), int(arr[1])
                rating_list.append([user, song])
        return rating_list

    def load_negative_file(self, filename):
        """Load the test negative file as a list of negative samples per user."""
        negative_list = []
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split("\t")
                int(arr[0])
                negatives = list(map(int, arr[1:])) # Convert negative songs  to integers
                negative_list.append(negatives)
        return negative_list

    def load_rating_file_as_matrix(self, filename):
        """
        Read train.rating file and return a sparse matrix.
        Each row represents a user, each column represents a song.
        """
        num_users, num_items = 0, 0
        interactions = []

        # First pass: Get max user_id and song_id
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                interactions.append((u, i))

        # Convert list of (user, song) pairs into separate arrays
        row, col = zip(*interactions)  # Extract user (row) and song (col) indices

        # Create a NumPy array with ones (for implicit feedback)
        data = np.ones(len(row), dtype=np.float32)

        # Construct the sparse matrix in CSR format
        mat = sp.csr_matrix((data, (row, col)), shape=(num_users + 1, num_items + 1), dtype=np.float32)

        logging.info(f"Loaded {len(interactions)} interactions into sparse matrix.")
        return mat

    def get_train_instances(self, num_negatives):
        """
        Generates training data with negative sampling.

        :param num_negatives: Number of negative samples per positive instance
        :return: user_input, song_input, labels (1 for positive, 0 for negative)
        """
        user_input, song_input, labels = [], [], []
        num_users, num_songs = self.trainMatrix.shape

        for (u, s) in zip(*self.trainMatrix.nonzero()):
            # Positive instance
            user_input.append(u)
            song_input.append(s)
            labels.append(1)

            # Negative sampling
            for _ in range(num_negatives):
                neg_s = np.random.randint(num_songs) # Random song
                while self.trainMatrix[u, neg_s] != 0: # Ensure it's not an actual interaction
                    neg_s = np.random.randint(num_songs)
                user_input.append(u)
                song_input.append(neg_s)
                labels.append(0)

        return user_input, song_input, labels
