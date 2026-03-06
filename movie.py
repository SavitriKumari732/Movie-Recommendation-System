import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import requests

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")


# Page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Machine Learning Based Movie Recommender")

st.markdown("""
<style>

.main {
background: linear-gradient(135deg,#141E30,#243B55);
}

h1,h2,h3 {
color:#FFD700;
text-align:center;
}

.stButton>button {
background-color:#ff4b4b;
color:white;
border-radius:8px;
height:45px;
width:200px;
font-size:16px;
}

.css-1d391kg {
background-color:#111111;
}

</style>
""", unsafe_allow_html=True)

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

st.header("📊 Dataset Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Movies", movies.shape[0])
col2.metric("Total Ratings", ratings.shape[0])
col3.metric("Unique Users", ratings['userId'].nunique())


# Load trained model
model = pickle.load(open("knn_model.pkl", "rb"))
movie_user_matrix = pickle.load(open("movie_matrix.pkl", "rb"))

# Sidebar
st.sidebar.title("Project Info")
st.sidebar.write("Algorithm: KNN")
st.sidebar.write("Similarity: Cosine")
st.sidebar.write("Dataset: MovieLens")

# Dataset Overview
st.header("📁 Dataset Preview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Movies Dataset")
    st.dataframe(movies.head())

with col2:
    st.subheader("Ratings Dataset")
    st.dataframe(ratings.head())

# Rating Distribution Graph
st.header("⭐ Rating Distribution")

fig, ax = plt.subplots()

sns.countplot(x='rating', data=ratings, ax=ax)

ax.set_title("Rating Distribution")

st.pyplot(fig)

# RMSE Calculation
movie_mean = ratings.groupby('movieId')['rating'].mean()
predictions = ratings['movieId'].map(movie_mean)

rmse = sqrt(mean_squared_error(
    ratings['rating'],
    predictions
))

st.header("🤖 Model Evaluation")

col1, col2 = st.columns(2)

col1.metric("RMSE Score", round(rmse,3))
col2.metric("Algorithm", "KNN (Cosine)")

# Recommendation System
st.divider()
st.header("🎥 Get Movie Recommendations")

movie_titles = movies['title'].values

selected_movie = st.selectbox(
    "Select a Movie",
    movie_titles
)

def fetch_poster(movie_title):

    # Remove year from title
    movie_title = movie_title.split("(")[0].strip()

    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"

    data = requests.get(url).json()

    if data['results'] and data['results'][0]['poster_path']:

        poster_path = data['results'][0]['poster_path']

        full_path = "https://image.tmdb.org/t/p/w500" + poster_path

        return full_path

    else:
        return None

def recommend_movies(movie_name, n_recommendations=5):

    movie_data = movies[movies['title'] == movie_name]

    if len(movie_data) == 0:
        return []

    movie_id = movie_data['movieId'].values[0]

    if movie_id not in movie_user_matrix.index:
        return []

    movie_row = movie_user_matrix.loc[movie_id].values.reshape(1, -1)

    distances, indices = model.kneighbors(
        movie_row,
        n_neighbors=n_recommendations + 1
    )

    recommended_ids = movie_user_matrix.index[indices[0][1:]]

    recommended_movies = movies[
        movies['movieId'].isin(recommended_ids)
    ]

    return recommended_movies['title'].values


if st.button("Recommend Movies"):

    recommendations = recommend_movies(selected_movie)

    st.subheader("Recommended Movies")

    if len(recommendations) == 0:
        st.write("No recommendations found")

    else:
        cols = st.columns(len(recommendations))

        for i, movie in enumerate(recommendations):

            poster = fetch_poster(movie)

            with cols[i]:

                if poster:
                    st.image(poster)

                st.write(movie)