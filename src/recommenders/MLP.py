"""
Multi-Layer Perceptron (MLP) for Neural Collaborative Filtering (NCF).
Adapted for the Music Recommender System.

Based on the original NCF implementation by He Xiangnan et al. in WWW 2017.
"""

import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Flatten, Concatenate
from keras.regularizers import l2
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from keras.initializers import RandomNormal
from src.data.dataset import Dataset
from src.recommenders.evaluate import evaluate_model
import argparse
import os
from time import time

#################### Argument Parsing ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP for MusicRecSys.")
    parser.add_argument('--data_dir', nargs='?', default='NCF_data/',
                        help='Path to the dataset directory.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. The first layer size determines the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative samples per positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default= 'adam',
                        help ='Optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations.')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

#################### Model Definition ####################
def get_model(num_users, num_songs, layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0]):
    assert len(layers) == len(reg_layers)
    num_layers = len(layers)

    # Inputs
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    song_input = Input(shape=(1,), dtype='int32', name='song_input')

    # Embeddings
    user_embedding = Embedding(input_dim=num_users, output_dim=layers[0] // 2,
                               embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
                               embeddings_regularizer=l2(reg_layers[0]))(user_input)

    song_embedding = Embedding(input_dim=num_songs, output_dim=layers[0] // 2,
                               embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
                               embeddings_regularizer=l2(reg_layers[0]))(song_input)

    # Flatten
    user_latent = Flatten()(user_embedding)
    song_latent = Flatten()(song_embedding)

    # Concatenate user & song embeddings
    vector = Concatenate()([user_latent, song_latent])

    # MLP layers
    for idx in range(1, num_layers):
        vector = Dense(layers[idx], activation='relu', kernel_regularizer=l2(reg_layers[idx]))(vector)

    # Prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform')(vector)

    model = Model(inputs=[user_input, song_input], outputs=prediction)
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

        # Negative sampling: Select 'num_negatives random songs the user hasn't listened to
        for _ in range(num_negatives):
            neg_s = np.random.randint(num_songs) # Random song
            while (u, neg_s) in train:
                neg_s = np.random.randint(num_songs) # Ensure it's not an actual interaction
            user_input.append(u)
            song_input.append(neg_s)
            labels.append(0)

    return user_input, song_input, labels

def train_mlp():
    args = parse_args()
    dataset = Dataset(args.data_dir)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_songs = train.shape

    # Build the MLP model
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    model = get_model(num_users, num_songs, layers, reg_layers)

    optimizer_dict = {
        "adagrad": Adagrad,
        "adam": Adam,
        "rmsprop": RMSprop,
        "sgd": SGD
    }
    chosen_optimizer = optimizer_dict[args.learner.lower()](learning_rate=args.lr)
    model.compile(optimizer=chosen_optimizer, loss='binary_crossentropy')

    # Training loop
    for epoch in range(args.epochs):
        user_input, song_input, labels = get_train_instances(train, args.num_neg)
        model.fit(
            [np.array(user_input), np.array(song_input)],
            np.array(labels),
            batch_size=args.batch_size,
            epochs=1,
            verbose=0,
            shuffle=True
        )

        # Evaluate every "verbose" epochs
        if epoch % args.verbose == 0:
            hits, ndcgs = evaluate_model(model, testRatings, testNegatives, K=10, num_threads=1)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print(f"Epoch {epoch}: HR={hr:.4f}, NDCG={ndcg:.4f}")

    # Save model
    if args.out > 0:
        model_path = os.path.join(args.data_dir, f"MLP_{args.layers}.h5")
        model.save_weights(model_path)
        print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_mlp()