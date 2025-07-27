import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os
import urllib.request
import zipfile

# Function to download and extract MovieLens dataset
@st.cache_data
def download_movielens_data():
    """Download and extract MovieLens dataset if not already present"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    movies_path = os.path.join(data_dir, "movies.csv")
    
    # Check if data already exists
    if os.path.exists(movies_path):
        return movies_path
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download MovieLens dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    
    with st.spinner("Downloading MovieLens dataset... This may take a moment."):
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Move files from subdirectory to data directory
        extracted_dir = os.path.join(data_dir, "ml-latest-small")
        for file in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, file)
            dst = os.path.join(data_dir, file)
            if os.path.isfile(src):
                os.rename(src, dst)
        
        # Clean up
        os.rmdir(extracted_dir)
        os.remove(zip_path)
    
    return movies_path

# Download data and load movies
movies_path = download_movielens_data()
movies = pd.read_csv(movies_path)

movies['genres'] = movies['genres'].fillna('')

movies['genres'] = movies['genres'].str.replace('|', ' ').str.lower()

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genres'])

def get_recommendations(movie_title, num_recommendations=5):
    try:
        index = movies[movies['title'] == movie_title].index[0]
        
        movie_vector = genre_matrix.getrow(index)
        
        similarities = cosine_similarity(movie_vector, genre_matrix).flatten()

        similar_indices = similarities.argsort()[::-1][1:num_recommendations+1]

        recommendations = []
        for idx in similar_indices:
            recommendations.append((movies.iloc[idx]['title'], similarities[idx]))
        
        return recommendations
    except IndexError:
        return []

st.title("🎬 Content-Based Movie Recommender")
movie_list = movies['title'].sort_values().tolist()
selected_movie = st.selectbox("Pick a movie you like:", movie_list)

if st.button("Recommend Similar Movies"):
    recommendations = get_recommendations(selected_movie)
    
    if recommendations:
        st.subheader("You might also like:")
        for title, score in recommendations:
            st.write(f"⭐ {title} (Similarity: {score:.2f})")
    else:
        st.error("Movie not found or no recommendations available.")