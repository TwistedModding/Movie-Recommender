import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load movie data
movies = pd.read_csv("data/movies.csv")

# Fill missing genres
movies['genres'] = movies['genres'].fillna('')

# Convert genres to lowercase with spaces
movies['genres'] = movies['genres'].str.replace('|', ' ').str.lower()

# Vectorize genres
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genres'])

# Compute similarity matrix
similarity = cosine_similarity(genre_matrix)

# UI
st.title("🎬 Content-Based Movie Recommender")
movie_list = movies['title'].sort_values().tolist()
selected_movie = st.selectbox("Pick a movie you like:", movie_list)

if st.button("Recommend Similar Movies"):
    index = movies[movies['title'] == selected_movie].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]  # Skip itself
    st.subheader("You might also like:")
    for i, score in scores:
        st.write(f"⭐ {movies.iloc[i]['title']} (Similarity: {score:.2f})")