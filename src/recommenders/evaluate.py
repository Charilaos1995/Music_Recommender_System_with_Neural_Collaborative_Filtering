"""
evaluate.py

Evaluate the performance of Top-K music recommendation.

Evaluation Protocol: Leave-1-out evaluation
Metrics:
    - Hit Ratio (HR) @ K: Measures if the ground truth song is in the Top-K list.
    - Normalized Discounted Cumulative Gain (NDCG) @ K: Rewards correct recommendations ranked higher.
"""

import math
import heapq # Used for retrieving top-K songs
import numpy as np
import logging
from time import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_model(model, testRatings, testNegatives, K, batch_size=1000):
    """
    Evaluate the model's performance using batched predictions.

    This method computes predictions for all user-song candidate pairs using the trained model,
    and evaluates them using HR and NDCG.

    :param model: Trained recommendation model (GMF, MLP, or NeuMF)
    :param testRatings: List of [user, ground-truth song] pairs from test.rating
    :param testNegatives: List of negative song IDs per user (not interacted with)
    :param K: Number of top recommendations to consider
    :param batch_size: Size of prediction batches (for memory efficiency)

    :return: (hits, ndcgs) - lists of HR and NDCG scores for each test instance
    """
    hits, ndcgs = [], []
    all_user_inputs = []
    all_song_inputs = []
    candidate_list = []

    # Prepare all (user, candidate) pairs for evaluation
    for rating, negatives in zip(testRatings, testNegatives):
        user_id = rating[0]
        gt_song = rating[1]
        candidates = negatives.copy()
        candidates.append(gt_song)
        candidate_list.append(candidates)
        all_user_inputs.extend([user_id] * len(candidates))
        all_song_inputs.extend(candidates)

    # Convert to numpy arrays
    all_user_inputs = np.array(all_user_inputs, dtype='int32')
    all_song_inputs = np.array(all_song_inputs, dtype='int32')

    # Predict scores in a single batch
    start_time = time()
    predictions = model.predict([all_user_inputs, all_song_inputs], batch_size=batch_size, verbose=0)
    logging.info(f"Batched prediction completed in {time() - start_time:.2f} seconds.")

    # Compute HR and NDCG for each user
    index = 0
    for i, candidates in enumerate(candidate_list):
        size = len(candidates)
        preds = predictions[index: index + size]
        index += size

        # Map each candidate song ID to its prediction score
        candidate_score = {candidates[j]: preds[j] for j in range(size)}

        # Get Top-K songs
        top_k = heapq.nlargest(K, candidate_score, key=candidate_score.get)

        # Evaluate
        hr = get_hit_ratio(top_k, testRatings[i][1])
        ndcg = get_ndcg(top_k, testRatings[i][1])
        hits.append(hr)
        ndcgs.append(ndcg)

    return hits, ndcgs

def get_hit_ratio(ranklist, gt_song):
    """
    Hit Ratio@K

    Checks whether the ground-truth song is in the Top-K recommended list.

    :param ranklist: List of recommended song IDs
    :param gt_song: The actual song the user has interacted with (ground truth)
    :return: 1 if gt_song is in ranklist, else 0
    """
    return 1 if gt_song in ranklist else 0

def get_ndcg(ranklist, gt_song):
    """
    Normalized Discounted Cumulative Gain (NDCG@K).

    NDCG rewards correct recommendations ranked higher in the Top-K list.

    Formula:
    NDCG@K = log(2) / log(position + 2)

    :param ranklist: List of Top-K recommended songs
    :param gt_song: The actual song the user interacted with (ground truth)
    :return: NDCG score (float between 0 and 1, higher is better)
    """
    for i, song in enumerate(ranklist):
        if song == gt_song:
            return math.log(2) / math.log(i + 2)  # Logarithmic discounting
    return 0

def recall_at_k(hits):
    """
    Recall@K

    In this 1-positive-per-user setting, Recall@K is the average Hit Ratio.

    :param hits: List of binary values (1 = hit, 0 = miss)
    :return: Mean recall value
    """
    return np.mean(hits)
