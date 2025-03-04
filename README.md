# Music Recommender System with Neural Collaborative Filtering

## Overview

This project implements a **Music Recommender System** based on **Neural Collaborative Filtering (NCF)**, as described in the original paper by He et al. (WWW 20217):
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering](https://dl.acm.org/doi/10.1145/3038912.3052569). In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

The system provides personalized music recommendations by analyzing **user-song interactions** and applying **deep learning-based collaborative filtering** techniques.

The framework includes:
- **Generalized Matrix Factorization (GMF)** 
- **Multi-Layer Perceptron (MLP)**
- **Neural Matrix Factorization (NeuMF)** (a hybrid approach combining GMF and MLP)

The system is trained on a dataset of user-song interactions and evaluated using **Hit Ratio (HR@K) and Normalized Discounted Cumulative Gain (NDCG@K)**.

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
│── models/ # Saved trained models (.h5 files) 
│── NCF_data/ # Dataset files for training and evaluation 
│ ├── train.rating # User-song interaction training data
│ ├── test.rating # Test set with ground-truth song for each user 
│ ├── test.negative # Negative samples for evaluation 
│ ├── song_data.csv # Song metadata (Last.fm subset) 
│ ├── triplets_file.csv # User-song interaction data (Echo Nest subset)
│ ├── cleaned_merged_music_data.csv # Merged dataset of user interactions and song metadata 
│── src/ # Source code 
│ ├── data/ 
│ │ ├── dataset.py # Loads and processes the dataset 
│ ├── recommenders/ 
│ │ ├── evaluate.py # Evaluates model performance 
│ │ ├── GMF.py # Generalized Matrix Factorization model 
│ │ ├── MLP.py # Multi-Layer Perceptron model 
│ │ ├── NeuMF.py # Neural Matrix Factorization model 
│── tests/ # Unit tests for dataset and models 
│── README.md # Documentation 
```

---

## Components and Functionality

### **Dataset Processing (`dataset.py`)**
- Loads **train, test, and negative samples** for training and evaluation.
- Converts user-song interactions into **sparse matrices** for efficiency.
- Ensures dataset files exist before loading.

### **Evaluation (`evaluate.py`)**
- Implements **Leave-One-Out (LOO) evaluation**.
- Computes:
  - **Hit Ratio (HR@K)** → Measures if the actual song appears in the top-K recommended songs.
  - **Normalized Discounted Cumulative Gain (NDCG@K)** → Rewards correct recommendations that are ranked higher.
- Supports **multi-threaded evaluation** for speed optimization.

### **Generalized Matrix Factorization (GMF) (`GMF.py`)**
- Uses **embedding layers** for user-song interactions.
- Performs **element-wise multiplication** between embeddings.
- Uses **binary cross-entropy loss** for training.
- Supports different optimizers (**Adam, RMSprop, SGD, Adagrad**).

### **Multi-Layer Perceptron (MLP) (`MLP.py`)**
- A **deep neural network model** for collaborative filtering.
- Embeddings are **concatenated instead of multiplied**.
- Supports **configurable layer sized** (e.g., '[64, 32, 16, 8]').

### **Neural Matrix Factorization (NeuMF) (`NeuMF.py`)**
- Combines **GMF's multiplicative interactions** with **MLP's deep feature learning**.
- Supports **pretraining**, where GMF and MLP are trained separately before being merged.
- Uses **both GMF & MLP embeddings**, concatenating them before final prediction.

___

## Running the Models

Each model is trained using command-line arguments, which can be modified as needed.

### **Run GMF**
```shell
python -m src.recommenders.GMF --data_dir NCF_data --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --verbose 1 --out 1
```
### **Run MLP**
```shell
python -m src.recommenders.MLP --data_dir NCF_data --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --verbose 1 --out 1
```
### **Run NeuMF (Without Pretraining)**
```shell
python -m src.recommenders.NeuMF --data_dir NCF_data --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --verbose 1 --out 1
```
### **Run NeuMF (With Pretraining)**
```shell
python -m src.recommenders.NeuMF --data_dir NCF_data --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --verbose 1 --out 1 --mf_pretrain models/GMF_8.h5 --mlp_pretrain models/MLP_[64,32,16,8].h5
```

## Model Evaluation

Once trained, the models are evaluated using the following metrics:
- **Hit Ratio (HR@K)**: Measures how often the correct song appears in the top-K list.
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Rewards correctly ranked recommendations.

To evaluate a trained model:
```shell
python -m src.recommenders.evaluate --model models/NeuMF.h5 --data_dir NCF_data --top_k 10 --num_threads 4
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