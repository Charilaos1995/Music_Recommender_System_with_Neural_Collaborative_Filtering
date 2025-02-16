import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path
spotify_data_path = '../NCF_data/spotify_data.csv'

# Load the dataset
spotify_df = pd.read_csv(spotify_data_path)

# Helper function for visualizing distributions
def plot_distribution(df, column, title):
    """
    Plot the distribution of a numerical column.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, bins=30, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Helper function for top genres
def top_genres(df, top_n=10):
    """
    Get the top genres by count.
    """
    genre_counts = df['genre'].value_counts().head(top_n)
    genre_counts.plot(kind='bar', figsize=(10, 6), color='orange', edgecolor='black')
    plt.title("Top Genres")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Helper function for feature correlation
def plot_feature_correlation(df):
    """
    Plot a heatmap of correlations between numerical features.
    """
    numerical_features = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numerical_features.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Between Features")
    plt.show()

# Main EDA Workflow
if __name__ == "__main__":
    print("Dataset Overview:")
    print(spotify_df.info())

    # Distribution of numerical features
    for feature in ['danceability', 'energy', 'acousticness', 'tempo', 'popularity']:
        plot_distribution(spotify_df, feature, f"Distribution of {feature}")

    # Top genres by count
    top_genres(spotify_df)

    # Correlation between features
    plot_feature_correlation(spotify_df)
