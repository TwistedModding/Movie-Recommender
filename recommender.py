import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

movies = pd.read_csv("data/movies.csv")

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