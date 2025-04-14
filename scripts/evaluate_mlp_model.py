"""
evaluate_mlp_model.py

This script loads a pre-trained MLP recommendation model and evaluates its performance on test data
using Hit Ratio (HR), Normalized Discounted Cumulative Gain (NDCG), and Recall@K.
"""

import os
import numpy as np
import logging
import warnings
from time import time

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow INFO and WARNING logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from src.data.dataset import Dataset
from src.recommenders.evaluate import evaluate_model, recall_at_k
from src.recommenders.MLP import get_model


def main():
    """
    Loads a trained MLP model and evaluates its recommendation performance using HR@10, NDCG@10, and Recall@10.

    This function:
    - Loads the test and training data using the Dataset class.
    - Loads the pre-trained MLP model weights.
    - Runs batched Top-K evaluation using leave-one-out methodology.
    - Prints evaluation metrics along with the total evaluation time.
    """
    # Get absolute project root (one level up from 'scripts' directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Define model path and data path
    data_dir = os.path.join(project_root, "NCF_data")
    model_path = os.path.join(project_root, "models", "MLP_[64,32,16,8].h5")

    print(f"Data directory: {data_dir}")
    print(f"Loading model from: {model_path}...")

    # Load dataset and model
    dataset = Dataset(data_dir)
    num_users, num_songs = dataset.trainMatrix.shape
    model = get_model(num_users, num_songs, [64, 32, 16, 8], [0, 0, 0, 0])
    model.load_weights(model_path)

    # Evaluate model
    start = time()
    hits, ndcgs = evaluate_model(model, dataset.testRatings, dataset.testNegatives, K=10, batch_size=1000)
    duration = time() - start

    # Display evaluation metrics
    print("\nEvaluation Results:")
    print(f"Hit Ratio (HR@10):   {np.mean(hits):.4f}")
    print(f"NDCG@10:             {np.mean(ndcgs):.4f}")
    print(f"Recall@10:           {recall_at_k(hits):.4f}")
    print(f"\nEvaluation Time: {duration:.2f} seconds")


if __name__ == '__main__':
    main()