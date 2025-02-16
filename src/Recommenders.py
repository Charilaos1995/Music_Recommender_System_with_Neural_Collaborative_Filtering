import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix

class Recommender:
    def __init__(self):
        self.train_data = None
        self.sparse_matrix = None
        self.users = None
        self.songs = None
        self.song_artist_mapping = {}
        self.user_knn_model = None
        self.item_knn_model = None

    def create_train_test_split(self, data, test_size=0.2):
        """
        Split the dataset into training and testing sets while filtering low-activity users & rare songs.
        """
        # Count user interactions
        user_counts = data['user_id'].value_counts()
        song_counts = data.groupby(['title', 'artist_name']).size()

        # Keep only users with at least 5 interactions and songs with at least 10 plays
        data = data[data['user_id'].isin(user_counts[user_counts >= 5].index)]
        data = data[data.set_index(['title', 'artist_name']).index.isin(song_counts[song_counts >= 10].index)]

        # Keep a mapping of song titles to artist names
        self.song_artist_mapping = data[['title', 'artist_name']].drop_duplicates().set_index('title').to_dict()['artist_name']

        train, test = train_test_split(data, test_size=test_size, random_state=42)
        self.train_data = train
        return train, test

    def create_sparse_matrix(self):
        """
        Directly create a sparse user-item matrix from the training data using coo_matrix.
        """
        # Create index mappings for users and songs
        user_ids = self.train_data['user_id'].unique()
        song_titles = self.train_data['title'].unique()

        user_id_to_index = {user: idx for idx, user in enumerate(user_ids)}
        song_to_index = {song: idx for idx, song in enumerate(song_titles)}

        # Save these lists for later reference
        self.users = list(user_ids)
        self.songs = list(song_titles)

        # Map each row of the DataFrame to corresponding matrix indices
        rows = self.train_data['user_id'].map(user_id_to_index).values
        cols = self.train_data['title'].map(song_to_index).values
        data = self.train_data['listen_count'].values

        # Build the sparse matrix directly
        self.sparse_matrix = csr_matrix((data, (rows, cols)), shape=(len(self.users), len(self.songs)))

    def fit_knn_models(self, n_neighbors=10):
        """
        Fit KNN models for both user and item recommendations.
        """
        # Fit user-based KNN
        self.user_knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
        self.user_knn_model.fit(self.sparse_matrix)

        # Fit item-based KNN (transpose the matrix to treat songs as rows)
        self.item_knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
        self.item_knn_model.fit(self.sparse_matrix.T)

    def recommend_items(self, user_id, top_n=10):
        """
        Recommend items to a user using a pre-fitted KNN model.
        """
        if self.sparse_matrix is None or not hasattr(self, 'user_knn_model'):
            raise ValueError("Sparse matrix or KNN model not initialized. Call 'create_sparse_matrix()' and 'fit_knn_models()' first.")

        if user_id not in self.users:
            print(f"User {user_id} not found in training data.")
            return []

        user_index = self.users.index(user_id)

        # Retrieve neighbors using the pre-fitted model
        distances, indices = self.user_knn_model.kneighbors(self.sparse_matrix[user_index], n_neighbors=top_n + 1)

        # Exclude the user itself (first neighbor) and compute the mean listen count
        similar_users = indices.flatten()[1:]
        recommended_scores = np.array(self.sparse_matrix[similar_users].mean(axis=0)).flatten()

        # Sort songs by their aggregated scores in descending order
        top_indices = recommended_scores.argsort()[::-1][:top_n]
        recommended_songs = [self.songs[i] for i in top_indices]

        return [(song, self.song_artist_mapping.get(song, "Unknown Artist")) for song in recommended_songs]

    def recommend_similar_items(self, title, top_n=10):
        """
        Recommend similar items to a given song using a pre-fitted KNN model.
        """
        if self.sparse_matrix is None or not hasattr(self, 'item_knn_model'):
            raise ValueError("Sparse matrix or item KNN model not initialized. Call 'create_sparse_matrix()' and 'fit_knn_models()' first.")

        if title not in self.songs:
            print(f"Song '{title}' not found in training data.")
            return []

        song_index = self.songs.index(title)

        # Retrieve similar songs using the pre-fitted item KNN model
        distances, indices = self.item_knn_model.kneighbors(self.sparse_matrix.T[song_index], n_neighbors=top_n + 1)

        similar_song_indices = indices.flatten()[1:]  # Exclude the song itself
        return [(self.songs[i], self.song_artist_mapping.get(self.songs[i], "Unknown Artist")) for i in similar_song_indices]

if __name__ == "__main__":
    # Load the dataset
    music_data_path = '../NCF_data/cleaned_merged_music_data.csv'
    music_df = pd.read_csv(music_data_path)

    # Initialize the recommender
    recommender = Recommender()

    # Create train-test split
    train, test = recommender.create_train_test_split(music_df)

    # Create sparse matrix
    recommender.create_sparse_matrix()

    # Fit KNN models
    recommender.fit_knn_models()

    # Recommend items for a user
    user_id = "796fe9de63d991a31cb9344a778dbd79eea1e815"
    recommendations = recommender.recommend_items(user_id, top_n=5)
    print(f"Recommender songs for user {user_id}: {recommendations}")

    # Recommend similar items for a song
    title = "One Step Closer (Album Version)"
    similar_songs = recommender.recommend_similar_items(title, top_n=5)
    print(f"Songs similar to '{title}': {similar_songs}")
