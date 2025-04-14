"""
evaluate_itemcf.py

Evaluates the performance of the Item-based Collaborative Filtering recommender system
using leave-one-out evaluation and Top-K metrics (Hit Ratio, NDCG, and Recall).

Each test user is evaluated using a ground truth song and 99 sampled negative songs.
"""

import os
import numpy as np
import heapq
import logging
from time import time

from src.data.dataset import Dataset
from src.recommenders.item_cf import ItemCFRecommender
from src.recommenders.evaluate import get_hit_ratio, get_ndcg, recall_at_k

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_itemcf(recommender, testRatings, testNegatives, K=10):
    """
    Evaluate the Item-based Collaborative Filtering (ItemCF) model using
    leave-one-out evaluation with Top-K recommendation metrics (HR, NDCG).

    :param recommender: Trained ItemCFRecommender instance
    :param testRatings: List of [user_id, ground_truth_song] pairs (from test.rating)
    :param testNegatives: List of negative item IDs per user (from test.negative)
    :param K: Top-K cutoff for recommendation evaluation
    :return: Two lists: Hit Ratios and NDCGs for each user
    """
    hits, ndcgs = [], []

    # Evaluate for each test user
    for rating, negatives in zip(testRatings, testNegatives):
        user_id, gt_song = rating

        # Get the user's interactions from the training matrix
        user_interactions = recommender.train_matrix[user_id].toarray().flatten()

        # Identify all items the user has interacted with (non-zero entries)
        interacted_items = np.where(user_interactions > 0)[0]

        # Build the candidate list: ground truth + sampled negatives
        candidates = negatives.copy()
        candidates.append(gt_song)

        # Score each candidate item for the current user
        if len(interacted_items) == 0:
            # If user has no interactions, assign 0 to all candidates
            candidate_scores = {i: 0.0 for i in candidates}
        else:
            # Sum similarities between candidate and all items user has seen
            candidate_scores = {
                i: np.sum(recommender.item_similarity[i, interacted_items])
                for i in candidates
            }

        # Select the Top-K items with the highest scores
        top_k = heapq.nlargest(K, candidate_scores, key=candidate_scores.get)

        # Evaluate using metrics
        hits.append(get_hit_ratio(top_k, gt_song))
        ndcgs.append(get_ndcg(top_k, gt_song))

    return hits, ndcgs


def main():
    """
    Loads the dataset and the ItemCF recommender, runs the evaluation process,
    and prints summary metrics including HR@10, NDCG@10, Recall@10, and total evaluation time.
    """
    # Dynamically compute path to NCF_data folder based on script location
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "NCF_data")

    # Load preprocessed data and initialize the recommender
    dataset = Dataset(data_dir)
    recommender = ItemCFRecommender(dataset.trainMatrix)

    print("Evaluating ItemCF recommender...")
    start = time()

    # Run evaluation on test data
    hits, ndcgs = evaluate_itemcf(recommender, dataset.testRatings, dataset.testNegatives, K=10)
    duration = time() - start

    # Compute averages
    avg_hr = np.mean(hits)
    avg_ndcg = np.mean(ndcgs)
    avg_recall = recall_at_k(hits)

    # Print formatted results
    print("\nEvaluation Results (ItemCF):")
    print(f"Hit Ratio (HR@10):   {avg_hr:.4f}")
    print(f"NDCG@10:             {avg_ndcg:.4f}")
    print(f"Recall@10:           {avg_recall:.4f}")
    print(f"\nEvaluation Time: {duration:.2f} seconds\n")

if __name__ == '__main__':
    main()