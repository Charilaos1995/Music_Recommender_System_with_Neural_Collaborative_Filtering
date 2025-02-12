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
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "../../data")

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
        logging.info(f"Dataset loaded: {self.num_users} users, {self.num_items} songs.")

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
                user = int(arr[0])
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

        # Construct sparse matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        for user, song in interactions:
            mat[user, song] = 1.0 # Implicit feedback (binary: 1 for interaction)

        logging.info(f"Loaded {len(interactions)} interactions into sparse matrix.")
        return mat
