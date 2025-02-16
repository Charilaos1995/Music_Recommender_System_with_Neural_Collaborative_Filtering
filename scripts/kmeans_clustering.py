import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px

# Load the dataset
data_path = '../NCF_data/spotify_data.csv'
df_main = pd.read_csv(data_path)

# Filter on genre and numeric features
genre_features = ['mode', 'acousticness', 'danceability', 'duration_ms',
                  'energy', 'instrumentalness', 'liveness', 'loudness',
                  'speechiness', 'tempo', 'valence', 'popularity', 'key', 'time_signature']
genres = df_main[['genre'] + genre_features]

# Clustering by Genre
print("\nClustering by genre...")
cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=10, random_state=42))
])
cluster_pipeline.fit(genres.select_dtypes(np.number))
genres['cluster'] = cluster_pipeline.predict(genres.select_dtypes(np.number))

# Visualizing Clusters with t-SNE
print("\nVisualizing genre clusters with t-SNE...")
tsne_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('tsne', TSNE(n_components=2, random_state=42, verbose=0))
])
genre_embedding = tsne_pipeline.fit_transform(genres.select_dtypes(np.number))
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genre'] = genres['genre']
projection['cluster'] = genres['cluster']

fig1 = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genre'],
                  title="t-SNE Visualization of Genre Clusters")
fig1.show()

# Clustering by Songs
print("\nClustering by songs...")
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, random_state=42))
])
X = df_main[genre_features]
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
df_main['cluster_label'] = song_cluster_labels

# Visualizing Clusters with PCA
print("\nVisualizing song clusters with PCA...")
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['track_name'] = df_main['track_name']
projection['cluster'] = df_main['cluster_label']

fig2 = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'track_name'],
                  title="PCA Visualization of Song Clusters")
fig2.show()

# Save clustered dataset
df_main.to_csv('../NCF_data/spotify_clustered_data.csv', index=False)
print("\nClustered dataset saved to: ../NCF_data/spotify_clustered_data.csv")