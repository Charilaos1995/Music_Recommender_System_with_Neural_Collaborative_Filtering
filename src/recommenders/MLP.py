"""
Multi-Layer Perceptron (MLP) for Neural Collaborative Filtering (NCF).
Adapted for the Music Recommender System.

Based on the original NCF implementation by He Xiangnan et al. in WWW 2017.
"""

import argparse
import os
import ast

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Flatten, Concatenate
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal

from src.data.dataset import Dataset
from src.recommenders.evaluate import evaluate_model


#################### Argument Parsing ####################
def parse_args():
    """
    Parses command-line arguments for training the MLP model.

    This function defines and returns all hyperparameters and paths needed
    to train the MLP model using the Neural Collaborative Filtering approach
    on the music dataset.

    :return: An argparse.Namespace object containing all the arguments.
    """
    parser = argparse.ArgumentParser(description="Run MLP for MusicRecSys.")

    # Path to the dataset directory
    parser.add_argument('--data_dir', nargs='?', default='NCF_data/',
                        help='Path to the dataset directory.')
    # Number of training epochs
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    # Number of samples per gradient update
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    # Architecture of the MLP: defines size of each layer (as stringified list)
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. The first layer size determines the embedding size.")
    # Regularization for each layer (L2), must match number of layers
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer.")
    # Number of negative samples to generate for each positive interaction
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative samples per positive instance.')
    # Learning rate for the optimizer
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    # Choice of optimizer
    parser.add_argument('--learner', nargs='?', default= 'adam',
                        help ='Optimizer to use: adagrad | adam | rmsprop | sgd')
    # Controls frequency of outputting performance metrics
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations.')
    # Whether to save the model to disk
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

#################### Model Definition ####################
def get_model(num_users, num_songs, layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0]):
    """
    Builds the Multi-Layer Perceptron (MLP) model for Neural Collaborative Filtering.

    This model learns user-song interactions using learned embeddings and
    passes them through fully connected layers to predict a binary interaction score.

    :param num_users: (int) Number of unique users in the dataset
    :param num_songs: (int) Number of unique songs in the dataset
    :param layers: (list of int) Layer sizes of the MLP (first value is also embedding size * 2)
    :param reg_layers: (list of float) L2 regularization values for each dense layer
    :return: (tf.keras.Model) Compiled Keras model ready for training
    """
    assert len(layers) == len(reg_layers), "layers and reg_layers must be the same length"
    num_layers = len(layers)

    # Input placeholders for user and item
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    song_input = Input(shape=(1,), dtype='int32', name='song_input')

    # Embedding layers for users and songs
    # Each embedding maps a user or song ID to a dense vector of size layers[0] // 2
    user_embedding = Embedding(
        input_dim=num_users,
        output_dim=layers[0] // 2,
        embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0])
    )(user_input)

    song_embedding = Embedding(
        input_dim=num_songs,
        output_dim=layers[0] // 2,
        embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0])
    )(song_input)

    # Flatten embedding tensors from shape (batch_size, 1, embedding_dim) to (batch_size, embedding_dim)
    user_latent = Flatten()(user_embedding)
    song_latent = Flatten()(song_embedding)

    # Concatenate user and song latent vectors
    vector = Concatenate()([user_latent, song_latent])

    # Build MLP layers as defined in `layers` (excluding the first which is for embedding)
    for idx in range(1, num_layers):
        vector = Dense(
            units=layers[idx],
            activation='relu',
            kernel_regularizer=l2(reg_layers[idx])
        )(vector)

    # Final prediction layer: outputs a single score between 0 and 1 using sigmoid
    prediction = Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer='lecun_uniform'
    )(vector)

    # Define and return the Keras model
    model = Model(inputs=[user_input, song_input], outputs=prediction)
    return model

#################### Training & Evaluation ####################
def train_mlp():
    """
    Train the MLP model for music recommendation using Neural Collaborative Filtering (NCF).

    The function loads training and testing data, initializes the MLP model based on the provided
    arguments, trains the model using binary cross-entropy loss, evaluates it every few epochs, and
    saves the final trained model if required.
    """
    # Parse command-line arguments
    args = parse_args()

    # Convert string list arguments to actual Python lists
    layers = ast.literal_eval(args.layers)
    reg_layers = ast.literal_eval(args.reg_layers)

    # Load dataset
    dataset = Dataset(args.data_dir)
    train = dataset.trainMatrix
    testRatings = dataset.testRatings
    testNegatives = dataset.testNegatives
    num_users, num_songs = train.shape

    # Initialize the MLP model
    model = get_model(num_users, num_songs, layers, reg_layers)

    # Choose optimizer based on argument
    optimizer_dict = {
        "adagrad": Adagrad,
        "adam": Adam,
        "rmsprop": RMSprop,
        "sgd": SGD
    }
    chosen_optimizer = optimizer_dict[args.learner.lower()](learning_rate=args.lr)

    # Compile the model with binary cross-entropy loss for implicit feedback
    model.compile(optimizer=chosen_optimizer, loss='binary_crossentropy')

    # Training loop for each epoch
    for epoch in range(args.epochs):
        print(f"Running : {epoch}")

        # Generate training data (positive + negative samples)
        user_input, song_input, labels = dataset.get_train_instances(args.num_neg)

        # Train on the current epoch
        model.fit(
            [np.array(user_input), np.array(song_input)],
            np.array(labels),
            batch_size=args.batch_size,
            epochs=1,
            verbose=0,
            shuffle=True
        )

        # Evaluate the model every "verbose" epochs
        if (epoch + 1) % args.verbose == 0:
            print(f"Starting model evaluation for epoch: {epoch + 1} ")
            hits, ndcgs = evaluate_model(model, testRatings, testNegatives, K=10)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print(f"Epoch {epoch + 1}: HR={hr:.4f}, NDCG={ndcg:.4f}")

    # Save trained model weights if requested
    if args.out > 0:
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"MLP_{args.layers}.h5")
        model.save_weights(model_path)
        print(f"Model saved to {model_path}")

# Run training if script is executed directly
if __name__ == '__main__':
    train_mlp()