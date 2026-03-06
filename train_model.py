# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from math import sqrt

# Load Dataset
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("Movies Shape:", movies.shape)
print("Ratings Shape:", ratings.shape)

# Rating Distribution Plot
sns.countplot(x='rating', data=ratings)
plt.title("Rating Distribution")
plt.show()

# Train-Test Split
train_data, test_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

print("Train Shape:", train_data.shape)
print("Test Shape:", test_data.shape)

# Remove Movies with < 50 Ratings
movie_counts = train_data['movieId'].value_counts()

train_filtered = train_data[
    train_data['movieId'].isin(movie_counts[movie_counts >= 50].index)
]

print("Filtered Train Shape:", train_filtered.shape)

# Create User-Movie Matrix
movie_user_matrix = train_filtered.pivot_table(
    index='movieId',
    columns='userId',
    values='rating'
).fillna(0)

print("User-Movie Matrix Shape:", movie_user_matrix.shape)

# Convert to Sparse Matrix
movie_user_sparse = csr_matrix(movie_user_matrix.values)

# Train KNN Model
model = NearestNeighbors(
    metric='cosine',
    algorithm='brute'
)

model.fit(movie_user_sparse)

print("Model Training Completed!")

# Save Model Files
pickle.dump(model, open("knn_model.pkl", "wb"))
pickle.dump(movie_user_matrix, open("movie_matrix.pkl", "wb"))
pickle.dump(movies, open("movies.pkl", "wb"))

print("Model Saved Successfully!")

# Basic Evaluation
movie_mean = train_filtered.groupby('movieId')['rating'].mean()

test_filtered = test_data[
    test_data['movieId'].isin(movie_mean.index)
]

predictions = test_filtered['movieId'].map(movie_mean)

rmse = sqrt(mean_squared_error(
    test_filtered['rating'],
    predictions
))

print("\nModel Evaluation")
print("RMSE:", rmse)