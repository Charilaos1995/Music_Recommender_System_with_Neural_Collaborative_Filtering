# Music Recommender System: MLP vs. Item-Based Collaborative Filtering

## Overview

This project implements a **Music Recommender System** that compares two approaches for generating music recommendations:
- **MLP Recommender:** A deep learning-based model built using Neural Collaborative Filtering principles.
Adapted from the original paper by He et al. (WWW 20217): Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering](https://dl.acm.org/doi/10.1145/3038912.3052569). 
In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.
- **Item-Based Collaborative Filtering (ItemCF):** A traditional recommender based on the similarity between items.

Both models use the same preprocessed dataset derived from The Million Song Dataset (MSD). The evaluation is performed using standard metrics such as **Hit Ratio (HR@K)**, **NDCG@K**, and **Recall@K**.
---

## Dataset Information

The dataset used in this project is derived from a subset of **The Million Song Dataset (MSD)**, specifically:
- **Echo Nest Taste Profile Subset** (`triplets_file.csv`): Contains user-song interaction data, recording play counts.
- **Last.fm Subset** (`song_data.csv`): Contains metadata about songs, including artist, title, and other relevant information.

These two datasets were **merged and processed** into:
- **`cleaned_merged_music_data.csv`**: A cleaned version combining `triplets_file.csv` and `song_data.csv`, mapping **user interactions to song metadata**.

From this merged dataset, the following files were generated:
- **`train.rating`**: Training data for the recommender system.
- **`test.rating`**: Test set containing ground-truth songs for each user.
- **`test.negative`**: Negative samples used for evaluation.

Only `train.rating`, `test.rating`, and `test.negative` are required for training and evaluation. 
They are include in the repository under `NCF_data/`, while the raw datasets (`triplets_file.csv`, `song_data.csv`, and `cleaned_merged_music_data.csv`) were used for preprocessing.

---

## Project Structure

```
Music_Recommender_System/
├── models/
│   └── MLP_[64,32,16,8].h5         # Pretrained MLP model weights
├── NCF_data/
│   ├── train.rating
│   ├── test.rating
│   ├── test.negative
│   ├── user_idx2original.pkl
│   └── song_idx2original.pkl
├── scripts/
│   ├── preprocess_sampled_music_data.py
│   ├── prepare_ncf_data.py
│   ├── recommend_songs.py               # Main CLI recommender
│   ├── evaluate_mlp_model.py
│   ├── evaluate_itemcf.py
│   ├── dataset_analysis.py
│   └── inspect_model.py
├── src/
│   ├── data/
│   │   └── dataset.py                   # Data loader and negative sampling
│   └── recommenders/
│       ├── MLP.py                       # MLP model definition and training
│       ├── item_cf.py                   # Item-based CF recommender
│       └── evaluate.py                  # Evaluation metrics
├── tests/
│   ├── test_dataset.py
│   ├── test_evaluate.py
│   └── test_file_matching.py
├── requirements.txt
└── README.md
```

---

## Components and Functionality

### Dataset Processing (`src/data/dataset.py`)
- Loads **train**, **test**, and **negative** samples.
- Converts user-song interactions into a **sparse matrix** for efficient storage and computation.
- Validates that all necessary files are present.

### Recommendation Models

#### MLP Recommender (`src/recommenders/MLP.py`)
- Implements a deep neural network based on collaborative filtering principles.
- Uses **concatenated embeddings** for users and songs.
- Configurable MLP layer sizes and regularization parameters.

#### Item-Based Collaborative Filtering (`src/recommenders/item_cf.py`)
- Computes item-item cosine similarity from the training interaction matrix.
- Generates recommendations based on the similarity scores of items the user has interacted with.

### Evaluation (`src/recommenders/evaluate.py`)
- Implements **Leave-One-Out (LOO) evaluation**.
- Computes:
  - **Hit Ratio (HR@K):** Checks if the ground-truth item appears in the Top-K recommendations.
  - **Normalized Discounted Cumulative Gain (NDCG@K):** Rewards correct recommendations when they appear at higher ranks.
  - **Recall@K:** The fraction of positive instances correctly recommended.

### Interactive CLI (`scripts/recommend_songs.py`)
- Provides a user-friendly command-line interface to:
  - Preview sample users and their listening histories.
  - Select a user interactively.
  - Choose between the MLP model, ItemCF, or both.
  - Generate recommendations and optionally save them to a CSV file.
  - Optionally re-run recommendations multiple times in a single session.

---

## Running the System

### Preprocessing Data
Run the `preprocess_sampled_music_data.py` script to merge and sample the raw datasets (if needed).

### Generating Training Data
Execute the `prepare_ncf_data.py` script to generate:
- `train.rating`
- `test.rating`
- `test.negative`
- Pickle files for user and song mappings

### Running Recommendations
To launch the interactive recommendation CLI, run:
```shell
python -m scripts.recommend_songs

___


## Model Evaluation

Once trained, the models are evaluated using standard recommendation metrics:
- **Hit Ratio (HR@K)**: Measures how often the ground-truth song appears in the Top-K recommendations.
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Rewards correct recommendations when they appear at higher ranks.
- **Recall@K**: Since each test user has exactly one positive instance, recall@K is equivalent to HR@K.

To evaluate the MLP recommender, run:
```shell
python -m scripts.evaluate_mlp_model

To evaluate the Item-Based Collaborative Filtering recommender, run:
python -m scripts.evaluate_itemcf

Both scripts print the average HR@K, NDCG@K, and Recall@K along with the evaluation runtime.

```

## Dependencies and Setup

### **Installing Dependencies**
To install the required dependencies, run:
```shell
pip install -r requirements.txt
```

### Required Libraries
The project requires the following Python libraries:
- `tensorflow>=2.6.0` → For deep learning models (GMF, MLP, NeuMF)
- `numpy` → For numerical operations and embeddings
- `scipy` → For sparse matrix storage (used in `dataset.py`)
- `pandas` → For handling dataset files
- `h5py` → For saving and loading trained models (`.h5` format)
- `matplotlib` → For visualizing evaluation metrics (optional)
- `scikit-learn` → For preprocessing and model evaluation
- `tqdm` → For displaying progress bars during prediction/evaluation

### Virtual Environment (Optional)
It is recommended to run the project inside a virtual environment. To create and activate one:

#### On Windows (Git Bash)
```shell
python -m venv MusicRecSys
source MusicRecSys/Scripts/activate
pip install -r requirements.txt
```

#### On macOS/Linux
```shell
python -m venv MusicRecSys
source MusicRecSys/bin/activate
pip install -r requirements.txt
```
To deactivate the virtual environment:
```shell
deactivate
```

### Verifying Installation
To check if all the required packages are installed, run:
```shell
pip list
```