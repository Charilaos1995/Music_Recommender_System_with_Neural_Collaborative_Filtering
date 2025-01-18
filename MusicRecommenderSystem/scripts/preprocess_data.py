import pandas as pd
import os

# Change working directory to the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working directory set to:", os.getcwd())

# Step 1: Load the dataset
file_path = 'data/Last.fm_data.csv'
df = pd.read_csv(file_path)

# Step 2: Inspect the dataset
print("First few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Step 3: Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Preprocess the dataset
# Example: Converting 'Date' and 'Time' into a single datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d %b %Y %H:%M')

# Dropping unnecessary columns
df = df.drop(columns=['Date', 'Time'])

# Normalize text data
df['Artist'] = df['Artist'].str.strip().str.lower()
df['Track'] = df['Track'].str.strip().str.lower()

# Step 5: Save the cleaned dataset
cleaned_file_path = 'data/cleaned_Last.fm_data.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")

# Display final structure
print("\nCleaned Dataset Info:")
print(df.info())
