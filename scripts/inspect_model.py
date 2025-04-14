import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Flatten, Concatenate
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal

# -------- Model Configuration --------
model_path = os.path.join("models", "MLP_[64,32,16,8].h5")
layers = [64, 32, 16, 8]
reg_layers = [0, 0, 0, 0]

# Dummy values (just to build the model structure)
num_users = 76353
num_items = 10000

# -------- Rebuild MLP Architecture --------
user_input = Input(shape=(1,), dtype='int32', name='user_input')
item_input = Input(shape=(1,), dtype='int32', name='song_input')

user_embedding = Embedding(
    input_dim=num_users,
    output_dim=layers[0] // 2,
    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
    embeddings_regularizer=l2(reg_layers[0])
)(user_input)

item_embedding = Embedding(
    input_dim=num_items,
    output_dim=layers[0] // 2,
    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
    embeddings_regularizer=l2(reg_layers[0])
)(item_input)

user_latent = Flatten()(user_embedding)
item_latent = Flatten()(item_embedding)

vector = Concatenate()([user_latent, item_latent])

for i in range(1, len(layers)):
    vector = Dense(layers[i], activation='relu', kernel_regularizer=l2(reg_layers[i]))(vector)

prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform')(vector)

model = Model(inputs=[user_input, item_input], outputs=prediction)

# -------- Load Weights and Show Summary --------
model.load_weights(model_path)
model.summary()
