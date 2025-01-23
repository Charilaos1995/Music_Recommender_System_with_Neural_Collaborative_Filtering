import pandas as pd
import numpy as np
import os

# Change working directory to the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working directory set to:", os.getcwd())

# File paths
#users_path = 'data/triplets_file.csv'
#songs_path = 'data/song_data.csv'
#output_path = 'data/cleaned_merged_data.csv'

# Step1 : Loading the Dataset
users_df = pd.read_csv('data/triplets_file.csv')
#print(users_df.head())

songs_df = pd.read_csv('data/song_data.csv')
#print(songs_df.head())

music_df = pd.merge(users_df, songs_df.drop_duplicates(['song_id']), on='song_id', how='left')
pd.set_option('display.max_columns', None)
#print(music_df.head())

#print(len(users_df), len(songs_df))
#print(len(music_df))

# Step 2: Data preprocessing
# Combine title and artist name into a new feature
music_df['song'] = music_df['title'] + ' - ' + music_df['artist_name']
# Drop the 'title' and 'artist_name' columns
music_df = music_df.drop(['title', 'artist_name'], axis=1)
# Print the updated DataFrame
#print(music_df.head())

# Taking the top 10k samples for quick results
music_df = music_df.head(10000)

# Cumulative sum of listen count for each song
music_grouped = music_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
print(music_grouped.head())

grouped_sum = music_grouped['listen_count'].sum()
music_grouped['percentage'] = (music_grouped['listen_count'] / grouped_sum) * 100
print(music_grouped.sort_values(['listen_count', 'song'], ascending=[0,1]))