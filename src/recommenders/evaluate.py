"""
Evaluate the performance of Top-K music recommendation.

Evaluation Protocol: Leave-1-out evaluation
Metrics:
    - Hit Ratio (HR) @ K: Measures if the ground truth song is in the Top-K list.
    - Normalized Discounted Cumulative Gain (NDCG) @ K: Rewards correct recommendations ranked higher.

Adapted from: Xiangnan He et al. (SIGIR'16)
"""

import math
import heapq # Used for retrieving top-K songs
import multiprocessing
import numpy as np
from time import time

# Global variables shared across multiple processes for multi-threaded evaluation
_model = None # The trained model used for predictions
_testRatings = None # List of (user, ground-truth song) pairs
_testNegatives = None # List of negative samples per user
_K = None # Number of top recommendations to consider (e.g., K=10)

def evaluate_model(model, testRatings, testNegatives, K, num_threads):
    """
    Evaluate the model's performance using Top-K recommendation.

    :param model: Trained recommendation model (GMF, MLP, or NeuMF)
    :param testRatings: List of (user, ground-truth song) pairs from test.rating
    :param testNegatives: List of 99 negative samples per user from test.negative
    :param K: Top-K recommendations (e.g., K=10)
    :param num_threads: Number of parallel threads for evaluation (default=1)
    :return: Two lists containing HR and NDCG scores for each test user
    """
    global _model, _testRatings, _testNegatives, _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs = [], []

    # Use multiprocessing for faster evaluation if num_threads > 1
    if num_threads > 1:
        pool = multiprocessing.Pool(processes=num_threads)
        results = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()

        # Unpack results from multiprocessing
        hits = [r[0] for r in results]
        ndcgs = [r[1] for r in results]
        return hits, ndcgs

    # Single-threaded evaluation
    for idx in range(len(_testRatings)):
        hr, ndcg = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)

    return hits, ndcgs

def eval_one_rating(idx):
    """
    Evaluate the recommendation quality for a single user.

    Steps:
    1. Retrieve the user's ground-truth song (positive sample).
    2. Retrieve 99 negative samples (songs the user has not interacted with).
    3. Compute predicted scores for all 100 songs (99 negatives + 1 ground truth).
    4. Sort predictions to get the Top-K recommended songs.
    5. Compute Hit Ration (HR) and NDCG for this user.

    :param idx: Index of the user in the test set
    :return: (HR, NDCG) score for this user
    """
    # Get the test user and their ground-truth song
    rating = _testRatings[idx]
    user_id = rating[0] # Extract user ID
    gt_song = rating[1] # Extract the correct song ID

    # Get the 99 negative song sampled for this user
    negatives = list(_testNegatives[idx])

    # Append the ground-truth song to the list of negatives for scoring
    negatives.append(gt_song)

    # Prepare inputs for model prediction (repeat user ID for all items)
    user_inputs = np.full(len(negatives), user_id, dtype='int32')
    song_inputs = np.array(negatives, dtype='int32')

    # Predict scores for all (user, song) pairs
    predictions = _model.predict([user_inputs, song_inputs], batch_size=100, verbose=0)

    # Store song scores in a dictionary
    song_score_map = {negatives[i]: predictions[i] for i in range(len(negatives))}

    # Remove the ground-truth song from the list (to maintain test integrity)
    negatives.pop()

    # Get the Top-K recommended songs
    top_k_songs = heapq.nlargest(_K, song_score_map, key=song_score_map.get)

    # Compute HR and NDCG for this user
    hr = get_hit_ratio(top_k_songs, gt_song)
    ndcg = get_ndcg(top_k_songs, gt_song)

    return hr, ndcg

def get_hit_ratio(ranklist, gt_song):
    """
    Compute Hit Ration (HR@K)

    HR@K = 1 if the ground-truth song appears in the Top-K recommendations, else 0.

    :param ranklist: List of Top-K recommended songs
    :param gt_song: The actual song the user has interacted with (ground truth)
    :return: 1 if gt_song is in ranklist, else 0
    """
    return 1 if gt_song in ranklist else 0

def get_ndcg(ranklist, gt_song):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG@K).

    NDCG rewards correct recommendations renked higher in the Top-K list.

    Formula:
    NDCG@K = log(2) / log(position + 2)

    :param ranklist: List of Top-K recommended songs
    :param gt_song: The actual song the user interacted with (ground truth)
    :return: NDCG score (higher is better)
    """
    for i in range(len(ranklist)):
        if ranklist[i] == gt_song:
            return math.log(2) / math.log(i + 2) # Logarithmic discounting
    return 0



