"""
Neural Matrix Factorization (NeuMF) model for the Music Recommender System.

This model combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
for hybrid collaborative filtering.

Adapted from original NCF implemented by He Xiangnan et al. in WWW 2017.

Features:
    - Supports both training from scratch and pretraining using GMF and MLP weights.
    - Integrates user-item interactions with learned embeddings.
    - Uses Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) for evaluation.
    - Optimized for TensorFlow/Keras with modern implementations.
"""

import argparse
import os
import ast
from time import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense, Concatenate, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop

from src.recommenders.GMF import get_model as get_gmf_model
from src.recommenders.MLP import get_model as get_mlp_model
from src.recommenders.evaluate import evaluate_model
from src.data.dataset import Dataset

# Argument parsing
def parse_args():
    """Parses command-line arguments for NeuMF training."""
    parser = argparse.ArgumentParser(description="Run NeuMF for Music Recommendation.")
    parser.add_argument('--data_dir', nargs='?', default='NCF_data/',
                        help='Path to the dataset directory.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of GMF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help='MLP layers.')
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for GMF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help='Regularization for each MLP layers.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative samples per positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--pretrain', type=bool, default=False,
                        help='User pretrained GMF & MLP weights.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Path to pretrained GMF model')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Path to pretrained MLP model.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X epochs.')
    parser.add_argument('--out', type=int, default=1,
                        help="Whether to save the trained model.")
    return parser.parse_args()

#################### Optimizer Selection ####################
def get_optimizer(learner, lr):
    """Returns the selected optimizer"""
    if learner.lower() == "adagrad":
        return Adagrad(learning_rate=lr)
    elif learner.lower() == "adam":
        return Adam(learning_rate=lr)
    elif learner.lower() == "rmsprop":
        return RMSprop(learning_rate=lr)
    else:
        return SGD(learning_rate=lr)

#################### Model Definition ####################
def get_neumf_model(num_users, num_songs, mf_dim=8, layers=[64, 32, 16, 8], reg_layers=[0,0,0,0], reg_mf=0):
    """Builds the NeuMF model by combining GMF and MLP components"""
    num_layers = len(layers)

    # Input layers
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    song_input = Input(shape=(1,), dtype='int32', name='song_input')

    # GMF part
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim,
                                  embeddings_regularizer=l2(reg_mf), name='mf_embedding_user')(user_input)
    MF_Embedding_Song = Embedding(input_dim=num_songs, output_dim=mf_dim,
                                  embeddings_regularizer=l2(reg_mf), name='mf_embedding_song')(song_input)

    mf_user_latent = Flatten()(MF_Embedding_User)
    mf_song_latent = Flatten()(MF_Embedding_Song)
    mf_vector = Multiply()([mf_user_latent, mf_song_latent])

    # MLP part
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0]//2,
                                   embeddings_regularizer=l2(reg_layers[0]), name='mlp_embedding_user')(user_input)
    MLP_Embedding_Song = Embedding(input_dim=num_songs, output_dim=layers[0]//2,
                                   embeddings_regularizer=l2(reg_layers[0]), name='mlp_embedding_song')(song_input)

    mlp_user_latent = Flatten()(MLP_Embedding_User)
    mlp_song_latent = Flatten()(MLP_Embedding_Song)
    mlp_vector = Concatenate()([mlp_user_latent, mlp_song_latent])

    for idx in range(1, num_layers):
        mlp_vector = Dense(layers[idx], activation='relu', kernel_regularizer=l2(reg_layers[idx]), name=f"layer{idx}")(mlp_vector)

    # NeuMF - Concatenating GMF and MLP outputs
    predict_vector = Concatenate()([mf_vector, mlp_vector])
    prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)

    model = Model(inputs=[user_input, song_input], outputs=prediction)
    return model

#################### Pretraining Integration ####################
def load_pretrain_model(neumf_model, gmf_pretrain_path, mlp_pretrain_path, layers):
    """Load pretrained GMF and MLP weights and integrate them into the NeuMF model."""
    # Retrieve the number of users and songs from the NeuMF embedding layers
    num_users = neumf_model.get_layer('mf_embedding_user').input_dim
    num_songs = neumf_model.get_layer('mf_embedding_song').input_dim
    mf_dim = neumf_model.get_layer('mf_embedding_user').output_dim

    # Load pretrained GMF model
    gmf_model = get_gmf_model(num_users, num_songs, mf_dim)
    gmf_model.load_weights(gmf_pretrain_path)

    # Load pretrained MLP model (using same layers configuration)
    mlp_model = get_mlp_model(num_users, num_songs, layers, reg_layers=[0]*len(layers))
    mlp_model.load_weights(mlp_pretrain_path)

    # Set weights for GMF embeddings
    neumf_model.get_layer('mf_embedding_user').set_weights(gmf_model.get_layer('user_embedding').get_weights())
    neumf_model.get_layer('mf_embedding_song').set_weights(gmf_model.get_layer('song_embedding').get_weights())

    # Set weights for MLP embeddings
    neumf_model.get_layer('mlp_embedding_user').set_weights(mlp_model.get_layer('user_embedding').get_weights())
    neumf_model.get_layer('mlp_embedding_song').set_weights(mlp_model.get_layer('song_embedding').get_weights())

    # Set weights for MLP layers
    num_layers = len(layers)
    for i in range(1, num_layers):
        neumf_model.get_layer(f"layer{i}").set_weights(mlp_model.get_layer(f"layer{i}").get_weights())

    # Combine prediction weights by averaging GMF and MLP prediction layers
    gmf_pred_weights = gmf_model.get_layer('prediction').get_weights()
    mlp_pred_weights = mlp_model.get_layer('prediction').get_weights()
    new_weights = 0.5 * np.concatenate((gmf_pred_weights[0], mlp_pred_weights[0]), axis=0)
    new_bias = 0.5 * (gmf_pred_weights[1] + mlp_pred_weights[1])
    neumf_model.get_layer('prediction').set_weights([new_weights, new_bias])

    return neumf_model

#################### Training with Periodic Evaluation ####################
def train_neumf(args):
    dataset = Dataset(args.data_dir)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_songs = train.shape

    layers = ast.literal_eval(args.layers)
    reg_layers = ast.literal_eval(args.reg_layers)

    # Build the NeuMF model
    neumf_model = get_neumf_model(num_users, num_songs, mf_dim=args.num_factors, layers=layers, reg_layers=reg_layers, reg_mf=args.reg_mf)
    optimizer = get_optimizer(args.learner, args.lr)
    neumf_model.compile(optimizer=optimizer, loss='binary_crossentropy')

    # Pretraining integration if specified
    if args.pretrain and args.mf_pretrain != '' and args.mlp_pretrain != '':
        print("Loading pretrained GMF and MLP models...")
        neumf_model = load_pretrain_model(neumf_model, args.mf_pretrain, args.mlp_pretrain, layers)
        print("Pretrained weights loaded.")

    best_hr, best_ndcg, best_epoch = 0, 0, -1

    # Training loop with periodic evaluation
    for epoch in range(args.epochs):
        t1 = time()
        # Generate training instances
        user_input, song_input, labels = dataset.get_train_instances(args.num_neg)
        # Train for one epoch
        neumf_model.fit([np.array(user_input), np.array(song_input)], np.array(labels),
                        batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Periodic evaluation every "verbose" epochs
        if epoch % args.verbose == 0:
            hits, ndcgs = evaluate_model(neumf_model, testRatings, testNegatives, K=10, num_threads=1)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            loss = neumf_model.evaluate([np.array(user_input), np.array(song_input)], np.array(labels), verbose=0)
            print(f"Epoch {epoch}: HR={hr:.4f}, NDCG={ndcg:.4f}, Loss={loss:.4f}, Time={(t2-t1):.1f}s")
            if hr > best_hr:
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                if args.out > 0:
                    os.makedirs("models", exist_ok=True)
                    model_out_file = os.path.join("models", f"NeuMF_{args.num_factors}_{args.layers}_{epoch}.h5")
                    neumf_model.save_weights(model_out_file)
                    print(f"Model saved to {model_out_file}.")

    print(f"End. Best Epoch {best_epoch}: HR={best_hr:.4f}, NDCG={best_ndcg:.4f}")
    if args.out > 0:
        os.makedirs("models", exist_ok=True)
        model_out_file = os.path.join("models", f"NeuMF_{args.num_factors}_{args.layers}_best.h5")
        print(f"Best model saved to {model_out_file}")

if __name__ == '__main__':
    args = parse_args()
    train_neumf(args)