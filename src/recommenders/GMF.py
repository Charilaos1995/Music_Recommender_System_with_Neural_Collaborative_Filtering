"""
Generalized Matrix Factorization (GMF) model for Neural Collaborative Filtering (NCF).
Adapted for the Music Recommender System.

Based on the original NCF implementation by He Xiangnan et al. in WWW 2017.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Flatten, Multiply
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from src.data.dataset import Dataset
from src.recommenders.evaluate import evaluate_model
import argparse
import os
from time import time

#################### Argument Parsing ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF for MusicRecSys.")
    parser.add_argument('--data_dir', nargs='?', default='NCF_data/',
                        help='Path to the dataset directory.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size (latent factors).')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and song embeddings.")
    parser.add_argument('--num_neg', type=int,default=4,
                        help='Number of negative samples per positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations.')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

#################### Model Definition ####################
def get_model(num_users, num_songs, latent_dim, regs=[0, 0]):
    """
    Builds the GMF model for user-song interactions.

    :param num_users: Number of uniques users in the dataset
    :param num_songs: Number of unique songs in the dataset
    :param latent_dim: Embedding size for users and songs
    :param regs: Regularization parameters for embedding
    :return: Compiled GMF model
    """
    # Input layers
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    song_input = Input(shape=(1,), dtype='int32', name='song_input')

    # Embedding layers
    user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim,
                               embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                               embeddings_regularizer=l2(regs[0]), input_length=1, name='user_embedding')

    song_embedding = Embedding(input_dim=num_songs, output_dim=latent_dim,
                               embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                               embeddings_regularizer=l2(regs[1]), input_length=1, name='song_embedding')

    # Flatten embeddings
    user_latent = Flatten()(user_embedding(user_input))
    song_latent = Flatten()(song_embedding(song_input))

    # Element-wise multiplication (GMF interaction)
    interaction = Multiply()([user_latent, song_latent])

    # Prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(interaction)

    # Build the model
    model = Model(inputs=[user_input, song_input], outputs=prediction)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

    return model

#################### Training & Evaluation ####################
def get_train_instances(train, num_negatives):
    """
    Generates training data with negative sampling.

    :param train: Sparse matrix of user-song interactions (train.rating)
    :param num_negatives: Number of negative samples per positive instance
    :return: user_input, song_input, labels (1 for positive, 0 for negative)
    """
    user_input, song_input, labels = [], [], []
    num_users, num_songs = train.shape

    for (u, s) in train.keys():
        # Positive instance
        user_input.append(u)
        song_input.append(s)
        labels.append(1)

        # Negative sampling: Select 'num_negatives' random songs the user hasn't listened to
        for _ in range(num_negatives):
            neg_s = np.random.randint(num_songs) # Random song
            while (u, neg_s) in train:
                neg_s = np.random.randint(num_songs) # Ensure it's not an actual interaction
            user_input.append(u)
            song_input.append(neg_s)
            labels.append(0)

    return user_input, song_input, labels


def train_gmf():
    args = parse_args()
    data_dir = args.data_dir

    # Load the dataset
    dataset = Dataset(data_dir)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_songs = train.shape

    print(f"Data Loaded: {num_users} users, {num_songs} songs.")

    # Get GMF model
    model = get_model(num_users, num_songs, args.num_factors, eval(args.regs))

    # Training loop
    for epoch in range(args.epochs):
        t1 = time()

        # Generate training instances
        user_input, song_input, labels = get_train_instances(train, args.num_neg)

        # Train the model
        hist = model.fit([np.array(user_input), np.array(song_input)], np.array(labels),
                         batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True)

        t2 = time()

        # Evaluate the model
        if epoch % args.verbose == 0:
            hits, ndcgs = evaluate_model(model, testRatings, testNegatives, K=10, num_threads=1)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            loss = hist.history['loss'][0]
            print(f"Epoch {epoch}: HR={hr:.4f}, NDCG={ndcg:.4f}, Loss{loss:.4f} [Time: {t2 - t1:.1f}s]")

    if args.out > 0:
        model_path = os.path.join(data_dir, f"GMF_{args.num_factors}.h5")
        model.save_weights(model_path)
        print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_gmf()
