import pandas as pd

# File path
spotify_data_path = '../data/spotify_data.csv'

# Load the dataset
spotify_df = pd.read_csv(spotify_data_path)

# Drop the unnamed first column
if 'Unnamed: 0' in spotify_df.columns:
    spotify_df = spotify_df.drop(columns=['Unnamed: 0'])
    print("Dropped the unnamed first column.")

# Display the shape of the dataset
print(f"\nDataset shape: {spotify_df.shape}")

# Define thr exploration function
def df_explore(df):
    """
    Explore the DataFrame for data types, missing values, and unique values.
    """
    missing = (pd.DataFrame((df.isna().sum() / df.shape[0]) * 100).reset_index().rename(columns={'index': 'column', 0: '%_missing'})
               .sort_values(by='%_missing', ascending=False))
    nunique = pd.DataFrame(df.nunique()).reset_index().rename(columns={'index': 'column', 0: 'nunique'}).sort_values(by='nunique', ascending=False)
    dtypes = pd.DataFrame(df.dtypes).reset_index().rename(columns={'index': 'column', 0: 'dtype'})

    return (pd.merge(pd.merge(dtypes, missing, on='column'), nunique, on='column', how='left').sort_values(by='%_missing', ascending=False)
            .sort_values(by='nunique', ascending=False))

# Explore the dataset
print("\nDataset exploration results:")
exploration_results = df_explore(spotify_df)
print(exploration_results)

# Drop rows with missing values
spotify_df = spotify_df.dropna(axis=0)
print("\nDropped rows with missing values.")
print(f"Updated dataset shape: {spotify_df.shape}")

# Step 5: Save the cleaned dataset
spotify_df.to_csv(spotify_data_path, index=False)
print(f"\nCleaned dataset saved to: {spotify_data_path}")