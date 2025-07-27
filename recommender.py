import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os
import urllib.request
import zipfile

class MovieRecommender:
    def __init__(self):
        self._load_data()
    
    def _download_movielens_data(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        movies_path = os.path.join(data_dir, "movies.csv")
        
        if os.path.exists(movies_path):
            return movies_path
        
        os.makedirs(data_dir, exist_ok=True)
        
        url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
        zip_path = os.path.join(data_dir, "ml-latest.zip")
        
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Move files from subdirectory and cleanup
        extracted_dir = os.path.join(data_dir, "ml-latest")
        for file in os.listdir(extracted_dir):
            os.rename(os.path.join(extracted_dir, file), os.path.join(data_dir, file))
        
        os.rmdir(extracted_dir)
        os.remove(zip_path)
        
        return movies_path
    
    def _load_data(self):
        movies_path = self._download_movielens_data()
        self.movies = pd.read_csv(movies_path)
        
        self.movies['year'] = self.movies['title'].str.extractall(r'\((\d{4})\)').groupby(level=0).last()
        self.movies['clean_title'] = self.movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
        self.movies['genres'] = self.movies['genres'].fillna('').str.replace('|', ' ').str.lower()
        
        self.vectorizer = TfidfVectorizer()
        self.genre_matrix = self.vectorizer.fit_transform(self.movies['genres'])
    
    def get_recommendations(self, movie_title, similarity_threshold=0.1, num_recommendations=5):
        try:
            index = self.movies[self.movies['title'] == movie_title].index[0]
        except IndexError:
            raise ValueError(f"Movie '{movie_title}' not found")
        
        movie_vector = self.genre_matrix.getrow(index)
        similarities = cosine_similarity(movie_vector, self.genre_matrix).flatten()
        
        all_indices = similarities.argsort()[::-1]
        
        recommendations = []
        for idx in all_indices:
            if idx == index:
                continue
                
            if similarities[idx] >= similarity_threshold:
                recommendations.append((self.movies.iloc[idx]['title'], similarities[idx]))
                if len(recommendations) >= num_recommendations:
                    break
        
        return recommendations
    
    def get_extended_recommendations(self, movie_title, similarity_threshold=0.1, start_index=5, batch_size=5):
        try:
            index = self.movies[self.movies['title'] == movie_title].index[0]
        except IndexError:
            raise ValueError(f"Movie '{movie_title}' not found")
        
        movie_vector = self.genre_matrix.getrow(index)
        similarities = cosine_similarity(movie_vector, self.genre_matrix).flatten()
        
        all_indices = similarities.argsort()[::-1]
        valid_recommendations = []
        
        for idx in all_indices:
            if idx == index:
                continue
                
            if similarities[idx] >= similarity_threshold:
                valid_recommendations.append((self.movies.iloc[idx]['title'], similarities[idx]))
        
        return valid_recommendations[start_index:start_index + batch_size]
    
    def get_available_years(self):
        return sorted(self.movies['year'].dropna().astype(int).unique(), reverse=True)
    
    def get_available_genres(self):
        all_genres = set()
        for genres in self.movies['genres'].str.split():
            all_genres.update(genres or [])
        return sorted(g for g in all_genres if g and g != '(no')
    
    def filter_movies(self, year=None, genre=None):
        filtered = self.movies.copy()
        
        if year and year != 'All Years':
            filtered = filtered[filtered['year'] == year]
        
        if genre and genre != 'All Genres':
            filtered = filtered[filtered['genres'].str.contains(genre, case=False, na=False)]
        
        return filtered

@st.cache_resource
def get_recommender():
    return MovieRecommender()

recommender = get_recommender()

st.title("Movie Recommender")

similarity_threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.3, 0.05)

col1, col2 = st.columns(2)

with col1:
    years = recommender.get_available_years()
    selected_year = st.selectbox("Filter by year:", ['All Years'] + [str(year) for year in years])

with col2:
    genres = recommender.get_available_genres()
    selected_genre = st.selectbox("Filter by genre:", ['All Genres'] + genres)

filtered_movies = recommender.filter_movies(selected_year, selected_genre)

if filtered_movies.empty:
    st.warning("No movies found with the selected filters.")
    selected_movie = None
else:
    # Show list with years
    movie_list = []
    for _, row in filtered_movies.iterrows():
        if pd.notna(row['year']):
            movie_list.append(f"{row['clean_title']} ({int(row['year'])})")
        else:
            movie_list.append(row['title'])
    
    selected_display = st.selectbox("Pick a movie:", sorted(movie_list))
    
    # Get original title
    if selected_display and '(' in selected_display:
        clean_title = selected_display.split(' (')[0]
        matches = recommender.movies[recommender.movies['clean_title'] == clean_title]
        selected_movie = matches.iloc[0]['title'] if not matches.empty else selected_display
    else:
        selected_movie = selected_display

if selected_movie and st.button("Get Recommendations"):
    st.session_state.recommendations_shown = 5
    st.session_state.current_movie = selected_movie
    st.session_state.similarity_threshold = similarity_threshold
    
    recommendations = recommender.get_recommendations(selected_movie, similarity_threshold)
    st.session_state.all_recommendations = recommendations or []

if st.session_state.get('all_recommendations'):
    st.subheader("Recommendations:")
    
    for title, score in st.session_state.all_recommendations:
        # Some display formatting
        if '(' in title and title.endswith(')'):
            st.write(f"• {title} ({score:.2f})")
        else:
            st.write(f"• {title} ({score:.2f})")
    
    # Show more button
    if (st.session_state.get('current_movie') and 
        st.session_state.get('recommendations_shown', 5) < 100):
        
        if st.button("Show More"):
            more_recs = recommender.get_extended_recommendations(
                st.session_state.current_movie,
                st.session_state.similarity_threshold,
                start_index=st.session_state.recommendations_shown,
                batch_size=5
            )
            
            if more_recs:
                st.session_state.all_recommendations.extend(more_recs)
                st.session_state.recommendations_shown += len(more_recs)
                st.rerun()
            else:
                st.info("No more recommendations available.")
                st.session_state.recommendations_shown = 100