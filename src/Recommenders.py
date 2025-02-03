import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class Recommender:
    def __init__(self):
        self.train_data = None
        self.user_item_matrix = None
        self.sparse_matrix = None
        self.users = None
        self.songs = None
        self.song_artist_mapping = {}

    def create_train_test_split(self, data, test_size=0.2):
        """
        Split the dataset into training and testing sets while filtering low-activity users & rare songs.
        """
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
        Create a sparse user-item interaction matrix based on listen counts.
        """
        self.user_item_matrix = self.train_data.pivot_table(
            index='user_id', columns='title', values='listen_count', aggfunc='sum', fill_value=0
        )

        self.users = self.user_item_matrix.index.tolist()
        self.songs = self.user_item_matrix.columns.tolist()

        # Store in class instead of only returning
        self.sparse_matrix = csr_matrix(self.user_item_matrix.values)

    def recommend_items(self, user_id, top_n=10):
        """
        Recommend items to a user using K-Nearest Neighbors for collaborative filtering.
        """
        if self.sparse_matrix is None:
            raise ValueError("Sparse matrix has not been created. Call 'create_sparse_matrix()' first.")

        if user_id not in self.users:
            print(f"User {user_id} not found in training data.")
            return []

        # Find user index
        user_index = self.users.index(user_id)

        # Use Nearest Neighbors to find similar users
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
        knn_model.fit(self.sparse_matrix)

        # Query model
        distances, indices = knn_model.kneighbors(self.sparse_matrix[user_index], n_neighbors=top_n + 1)

        # Get recommendations from similar users
        similar_users = indices.flatten()[1:]
        recommended_songs = self.user_item_matrix.iloc[similar_users].mean().sort_values(ascending=False).head(top_n)

        return [(title, self.song_artist_mapping.get(title, "Unknown Artist")) for title in recommended_songs.index]

    def recommend_similar_items(self, title, top_n=10):
        """
        Recommend similar items using KNN based on song similarity.
        """
        if self.sparse_matrix is None:
            raise ValueError("Sparse matrix has not been created. Call 'create_sparse_matrix()' first.")

        if title not in self.songs:
            print(f"Song '{title}' not found in training data.")
            return []

        # Find song index
        song_index = self.songs.index(title)

        # Use Nearest Neighbors to find similar songs
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=top_n + 1)
        knn_model.fit(self.sparse_matrix.T)

        distances, indices = knn_model.kneighbors(self.sparse_matrix.T[song_index], n_neighbors=top_n + 1)

        # Get recommended songs
        similar_songs = indices.flatten()[1:]
        return [(self.songs[i], self.song_artist_mapping.get(self.songs[i], "Unknown Artist")) for i in similar_songs]


if __name__ == "__main__":
    # Load the dataset
    music_data_path = '../data/cleaned_merged_music_data.csv'
    music_df = pd.read_csv(music_data_path)

    # Initialize the recommender
    recommender = Recommender()

    # Create train-test split
    train, test = recommender.create_train_test_split(music_df)

    # Create sparse matrix before making recommendations
    recommender.create_sparse_matrix()

    # Recommend items for a user
    user_id = "969cc6fb74e076a68e36a04409cb9d3765757508"
    recommendations = recommender.recommend_items(user_id, top_n=5)
    print(f"Recommender songs for user {user_id}: {recommendations}")

    # Recommend similar items for a song
    title = "Mr. Jones"
    similar_songs = recommender.recommend_similar_items(title, top_n=5)
    print(f"Songs similar to '{title}': {similar_songs}")
