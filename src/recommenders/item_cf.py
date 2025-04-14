"""
Item-based Collaborative Filtering Recommender

This module defines a class for computing item-based collaborative filtering
recommendations using cosine similarity between items.
Designed to be used as a baseline model for comparison with the more advanced MLP model.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class ItemCFRecommender:
    def __init__(self, train_matrix: csr_matrix):
        """
        Initializes the Item-based Collaborative Filtering recommender.

        :param train_matrix: Sparse user-item interaction matrix (CSR format).
                             Shape: (num_users), (num_items)
        """
        self.train_matrix = train_matrix # Store the training interaction matrix

        # Compute item-to-item cosine similarity matrix
        # Transpose is used so we compute similarity across items (columns)
        self.item_similarity = cosine_similarity(train_matrix.T)

        # Set self-similarity to 0 (avoid influencing recommendations with the same item)
        np.fill_diagonal(self.item_similarity, 0)

    def predict(self, inputs, batch_size=100, verbose=0):
        """
        Predict scores for given (user, item) pairs using item-based collaborative filtering.

        :param inputs: Tuple of (user_ids, item_ids)
                       - user_ids: array of user indices
                       - item_ids: array of item indices to score for each user
        :param batch_size: Unused in this implementation. Present for compatibility.
        :param verbose: Verbosity level (0 = silent)

        :return: NumPy array of predicted relevance scores for each (user, item) pair.
        """
        user_inputs, item_inputs = inputs
        scores = []

        for u, i in zip(user_inputs, item_inputs):
            # Get binary vector of items the user has interacted with
            user_interactions = self.train_matrix[u].toarray().flatten() # binary vector of length num_items
            interacted_items = np.where(user_interactions > 0)[0]

            if len(interacted_items) == 0:
                # If no interactions exist for the user, return score of 0
                scores.append(0.0)
            else:
                # Compute score as sum of similarities between candidate item and previously interacted items
                score = np.sum(self.item_similarity[i, interacted_items])
                scores.append(score)

        return np.array(scores)


