import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os
import urllib.request
import zipfile
from typing import List, Tuple, Optional

class MovieRecommender:
    """A content-based movie recommendation system using TF-IDF and cosine similarity."""
    
    def __init__(self):
        self.movies: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.genre_matrix = None
        self._load_data()
    
    @st.cache_data
    def _download_movielens_data(_self):
        """Download and extract MovieLens dataset if not already present"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        movies_path = os.path.join(data_dir, "movies.csv")
        
        if os.path.exists(movies_path):
            return movies_path
        
        os.makedirs(data_dir, exist_ok=True)
        
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_path = os.path.join(data_dir, "ml-latest-small.zip")
        
        with st.spinner("Downloading MovieLens dataset... This may take a moment."):
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            extracted_dir = os.path.join(data_dir, "ml-latest-small")
            for file in os.listdir(extracted_dir):
                src = os.path.join(extracted_dir, file)
                dst = os.path.join(data_dir, file)
                if os.path.isfile(src):
                    os.rename(src, dst)
            
            os.rmdir(extracted_dir)
            os.remove(zip_path)
        
        return movies_path
    
    def _load_data(self):
        """Load and preprocess movie data"""
        movies_path = self._download_movielens_data()
        self.movies = pd.read_csv(movies_path)
        
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)$')[0]
        self.movies['clean_title'] = self.movies['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        self.movies['genres'] = self.movies['genres'].fillna('')
        self.movies['genres'] = self.movies['genres'].str.replace('|', ' ').str.lower()
        
        self.vectorizer = TfidfVectorizer()
        self.genre_matrix = self.vectorizer.fit_transform(self.movies['genres'])
    
    def get_recommendations(self, movie_title: str, similarity_threshold: float = 0.1, 
                          num_recommendations: int = 5) -> List[Tuple[str, float]]:
        """Get movie recommendations based on similarity threshold"""
        if self.movies is None or self.genre_matrix is None:
            return []
            
        try:
            index = self.movies[self.movies['title'] == movie_title].index[0]
            
            movie_vector = self.genre_matrix.getrow(index)
            similarities = cosine_similarity(movie_vector, self.genre_matrix).flatten()
            
            valid_indices = similarities.argsort()[::-1][1:]
            
            recommendations = []
            for idx in valid_indices:
                if similarities[idx] >= similarity_threshold:
                    recommendations.append((self.movies.iloc[idx]['title'], similarities[idx]))
                    if len(recommendations) >= num_recommendations:
                        break
            
            return recommendations
        except IndexError:
            return []
    
    def get_extended_recommendations(self, movie_title: str, similarity_threshold: float = 0.1, 
                                   start_index: int = 5, batch_size: int = 5) -> List[Tuple[str, float]]:
        """Get additional recommendations starting from a specific index"""
        if self.movies is None or self.genre_matrix is None:
            return []
            
        try:
            index = self.movies[self.movies['title'] == movie_title].index[0]
            
            movie_vector = self.genre_matrix.getrow(index)
            similarities = cosine_similarity(movie_vector, self.genre_matrix).flatten()
            
            all_similar_indices = similarities.argsort()[::-1][1:]
            
            valid_recommendations = []
            for idx in all_similar_indices:
                if similarities[idx] >= similarity_threshold:
                    valid_recommendations.append((self.movies.iloc[idx]['title'], similarities[idx]))
            
            end_index = start_index + batch_size
            return valid_recommendations[start_index:end_index]
        except (IndexError, ValueError):
            return []
    
    def get_available_years(self) -> List[int]:
        """Get sorted list of available years"""
        if self.movies is None:
            return []
        return sorted(self.movies['year'].dropna().unique().astype(int), reverse=True)
    
    def get_available_genres(self) -> List[str]:
        """Get sorted list of available genres"""
        if self.movies is None:
            return []
        all_genres = set()
        for genre_list in self.movies['genres'].str.split():
            if isinstance(genre_list, list):
                all_genres.update(genre_list)
        return sorted([g for g in all_genres if g and g != '(no'])
    
    def filter_movies(self, year: Optional[str] = None, genre: Optional[str] = None) -> pd.DataFrame:
        """Filter movies by year and/or genre"""
        if self.movies is None:
            return pd.DataFrame()
            
        filtered_movies = self.movies.copy()
        
        if year and year != 'All Years':
            filtered_movies = filtered_movies[filtered_movies['year'] == year]
        
        if genre and genre != 'All Genres':
            filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(genre, case=False, na=False)]
        
        return filtered_movies

@st.cache_resource
def get_recommender():
    return MovieRecommender()

def main():
    recommender = get_recommender()
    
    st.title("Content-Based Movie Recommender")

    st.subheader("Recommendation Settings")
    similarity_threshold = st.slider(
        "Minimum similarity threshold:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.8, 
        step=0.05,
        help="Higher values = more similar movies (but fewer results). Lower values = more variety (but more results)."
    )
    st.write(f"*Showing movies with at least {similarity_threshold:.0%} similarity*")

    col1, col2 = st.columns(2)

    with col1:
        years = recommender.get_available_years()
        selected_year = st.selectbox("Filter by year:", 
                                     options=['All Years'] + [str(year) for year in years])

    with col2:
        genres = recommender.get_available_genres()
        selected_genre = st.selectbox("Filter by genre:", 
                                      options=['All Genres'] + genres)

    filtered_movies = recommender.filter_movies(selected_year, selected_genre)

    if len(filtered_movies) > 0:
        movie_display_list = []
        for _, row in filtered_movies.iterrows():
            if pd.notna(row['year']):
                display_title = f"{row['clean_title']} ({int(float(row['year']))})"
            else:
                display_title = row['title']
            movie_display_list.append(display_title)
        
        movie_display_list = sorted(movie_display_list)
        selected_movie_display = st.selectbox("Pick a movie you like:", movie_display_list)
        
        if selected_movie_display and '(' in selected_movie_display and selected_movie_display.endswith(')'):
            selected_movie_title = selected_movie_display.rsplit(' (', 1)[0]
            if recommender.movies is not None:
                matching_movies = recommender.movies[recommender.movies['clean_title'] == selected_movie_title]
                if len(matching_movies) > 0:
                    selected_movie = matching_movies.iloc[0]['title']
                else:
                    selected_movie = selected_movie_display
            else:
                selected_movie = selected_movie_display
        else:
            selected_movie = selected_movie_display
    else:
        st.warning("No movies found with the selected filters.")
        selected_movie = None

    if selected_movie and st.button("Recommend Similar Movies"):
        st.session_state.recommendations_shown = 5
        st.session_state.current_movie = selected_movie
        st.session_state.similarity_threshold = similarity_threshold
        st.session_state.all_recommendations = []
        
        recommendations = recommender.get_recommendations(selected_movie, similarity_threshold)
        
        if recommendations:
            st.session_state.all_recommendations = recommendations
        else:
            st.error("Movie not found or no recommendations available with the selected similarity threshold. Try lowering the threshold.")

    if hasattr(st.session_state, 'all_recommendations') and st.session_state.all_recommendations:
        st.subheader("You might also like:")
        
        for i, (title, score) in enumerate(st.session_state.all_recommendations):
            display_title = title
            year_match = pd.Series([title]).str.extract(r'\((\d{4})\)$')[0].iloc[0]
            if pd.notna(year_match):
                clean_title = title.replace(f' ({year_match})', '')
                display_title = f"{clean_title} ({year_match})"
            
            st.write(f"{display_title} (Similarity: {score:.2f})")
        
        current_movie = getattr(st.session_state, 'current_movie', None)
        stored_threshold = getattr(st.session_state, 'similarity_threshold', 0.1)
        recommendations_shown = getattr(st.session_state, 'recommendations_shown', 5)
        
        if current_movie and recommendations_shown < 100:
            st.write(f"*Showing {len(st.session_state.all_recommendations)} recommendations (≥{stored_threshold:.0%} similarity)*")
            if st.button("🔍 Show More Recommendations"):
                more_recommendations = recommender.get_extended_recommendations(
                    current_movie, 
                    stored_threshold,
                    start_index=recommendations_shown, 
                    batch_size=5
                )
                
                if more_recommendations:
                    st.session_state.all_recommendations.extend(more_recommendations)
                    st.session_state.recommendations_shown += len(more_recommendations)
                    st.rerun()
                else:
                    st.info("No more recommendations available with the selected similarity threshold.")
                    st.session_state.recommendations_shown = 100
        elif current_movie and recommendations_shown >= 100:
            st.write(f"*Showing {len(st.session_state.all_recommendations)} recommendations (maximum reached)*")
            st.info("Reached maximum number of recommendations (100). Try a different movie for more suggestions!")

if __name__ == "__main__":
    main()