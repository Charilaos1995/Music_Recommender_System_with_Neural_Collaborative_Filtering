import sys
import os
import numpy as np
import tensorflow as tf

from src.recommenders.evaluate import evaluate_model
from src.recommenders.GMF import get_model  # Use GMF as a simple test model
from src.data.dataset import Dataset

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def test_evaluate():
    """Test the evaluate.py script with a simple GMF model"""

    print("\nRunning evaluation test...")

    # Load dataset
    dataset = Dataset()
    testRatings, testNegatives = dataset.testRatings, dataset.testNegatives
    num_users, num_songs = dataset.num_users, dataset.num_items

    # Build a simple untrained GMF model for testing
    model = get_model(num_users, num_songs, latent_dim=8)

    # Run evaluation
    print("\nEvaluating with random weights...")
    hits, ndcgs = evaluate_model(model, testRatings, testNegatives, K=10, num_threads=1)

    # Print summary
    print(f"\nEvaluation completed!")
    print(f"Average HR@10: {np.mean(hits):.4f}")
    print(f"Average NDCG@10: {np.mean(ndcgs):.4f}")

if __name__ == "__main__":
    test_evaluate()
