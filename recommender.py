import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
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
        
        # clean up
        extracted_dir = os.path.join(data_dir, "ml-latest")
        for file in os.listdir(extracted_dir):
            os.rename(os.path.join(extracted_dir, file), os.path.join(data_dir, file))
        
        os.rmdir(extracted_dir)
        os.remove(zip_path)
        
        return movies_path
    
    def _load_data(self):
        movies_path = self._download_movielens_data()
        self.movies = pd.read_csv(movies_path)
        
        ratings_path = os.path.join(os.path.dirname(movies_path), "ratings.csv")
        
        # chunk loading and filtering for memory
        print("Loading chunks...")
        chunk_size = 1000000
        filtered_chunks = []
        
        movie_counts = {}
        total_processed = 0
        
        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size):
            total_processed += len(chunk)
            print(f"Processing chunk... {total_processed:,}")
            
            chunk_counts = chunk['movieId'].value_counts()
            for movie_id, count in chunk_counts.items():
                movie_counts[movie_id] = movie_counts.get(movie_id, 0) + count
        
        popular_movies = pd.Series(movie_counts).nlargest(2000).index.tolist()
        print(f"Selected top {len(popular_movies)} popular movies from {len(movie_counts)} total movies")
        
        total_processed = 0
        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size):
            total_processed += len(chunk)
            print(f"Filtering chunk... Total rows processed: {total_processed:,}")
            
            filtered_chunk = chunk[chunk['movieId'].isin(popular_movies)]
            if len(filtered_chunk) > 0:
                filtered_chunks.append(filtered_chunk)
        
        if filtered_chunks:
            self.ratings = pd.concat(filtered_chunks, ignore_index=True)
            print(f"Final ratings dataset: {len(self.ratings):,} ratings for {self.ratings['movieId'].nunique()} movies")
        else:
            print("No ratings data found, using empty dataset")
            self.ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
        
        # get year and clean title
        self.movies['year'] = self.movies['title'].str.extractall(r'\((\d{4})\)').groupby(level=0).last()
        self.movies['clean_title'] = self.movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
        
        self.movies['genres'] = self.movies['genres'].fillna('').str.replace('|', ' ').str.lower()
        self.vectorizer = TfidfVectorizer()
        self.genre_matrix = self.vectorizer.fit_transform(self.movies['genres'])
        
        self._build_collaborative_model()
        
        self._build_hybrid_model()
    
    def _build_collaborative_model(self):
        movie_counts = self.ratings['movieId'].value_counts()
        user_counts = self.ratings['userId'].value_counts()
        
        popular_movies = movie_counts.head(1000).index
        active_users = user_counts.head(2000).index
        
        filtered_ratings = self.ratings[
            (self.ratings['movieId'].isin(popular_movies)) & 
            (self.ratings['userId'].isin(active_users))
        ]
        
        print(f"Filtered from {len(self.ratings)} to {len(filtered_ratings)} ratings")
        print(f"Movies: {len(popular_movies)}, Users: {len(active_users)}")
        print(f"Matrix size: {len(active_users)} x {len(popular_movies)} = {len(active_users) * len(popular_movies):,} cells")
        
        if len(filtered_ratings) < 1000:
            print("Insufficient data for collaborative filtering, using content-based only")
            self.movieid_to_idx = {}
            self.idx_to_movieid = {}
            return
        
        user_movie_matrix = filtered_ratings.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        print(f"Actual matrix shape: {user_movie_matrix.shape}")
        
        n_components = min(30, min(user_movie_matrix.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(user_movie_matrix)
        self.movie_factors = self.svd.components_.T
        
        print(f"SVD completed with {n_components} components")
        
        self.movieid_to_idx = {}
        self.idx_to_movieid = {}
        
        for idx, row in self.movies.iterrows():
            movie_id = row['movieId']
            if movie_id in user_movie_matrix.columns:
                col_idx = user_movie_matrix.columns.get_loc(movie_id)
                self.movieid_to_idx[movie_id] = col_idx
                self.idx_to_movieid[col_idx] = movie_id
    
    def _build_hybrid_model(self):
        """Prepare hybrid model without precomputing large matrices"""
        if not hasattr(self, 'movie_factors') or len(self.movieid_to_idx) == 0:
            print("Using content-based filtering only (collaborative model unavailable)")
            self.use_hybrid = False
        else:
            print(f"Hybrid model ready with {len(self.movieid_to_idx)} movies having collaborative data")
            self.use_hybrid = True
            self.collab_similarities = cosine_similarity(self.movie_factors)
    
    def get_recommendations(self, movie_title, similarity_threshold=0.1, num_recommendations=5, use_ai=True):
        try:
            index = self.movies[self.movies['title'] == movie_title].index[0]
        except IndexError:
            raise ValueError(f"Movie '{movie_title}' not found")
        
        if use_ai and hasattr(self, 'use_hybrid') and self.use_hybrid:
            similarities = self._compute_hybrid_similarities(index)
        else:
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
    
    def _compute_hybrid_similarities(self, movie_index):
        """Compute hybrid similarities for a single movie on-demand"""
        movie_id = self.movies.iloc[movie_index]['movieId']
        
        movie_vector = self.genre_matrix.getrow(movie_index)
        content_sims = cosine_similarity(movie_vector, self.genre_matrix).flatten()
        
        if movie_id in self.movieid_to_idx:
            collab_idx = self.movieid_to_idx[movie_id]
            collab_sims = self.collab_similarities[collab_idx]
            
            hybrid_sims = content_sims.copy()
            
            for i, (mid, col_idx) in enumerate(self.movieid_to_idx.items()):
                movie_match = self.movies[self.movies['movieId'] == mid]
                if not movie_match.empty:
                    full_idx = movie_match.index[0]
                    hybrid_sims[full_idx] = 0.6 * collab_sims[col_idx] + 0.4 * content_sims[full_idx]
            
            return hybrid_sims
        else:
            return content_sims
    
    def get_extended_recommendations(self, movie_title, similarity_threshold=0.1, start_index=5, batch_size=5, use_ai=True):
        try:
            index = self.movies[self.movies['title'] == movie_title].index[0]
        except IndexError:
            raise ValueError(f"Movie '{movie_title}' not found")
        
        if use_ai and hasattr(self, 'use_hybrid') and self.use_hybrid:
            similarities = self._compute_hybrid_similarities(index)
        else:
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
    
    def get_popular_recommendations(self, num_recommendations=10):
        """Get AI-powered popular recommendations based on ratings"""
        if not hasattr(self, 'ratings'):
            return []
        
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        movie_stats.columns = ['rating_count', 'rating_avg']
        movie_stats = movie_stats.reset_index()
        
        min_ratings = movie_stats['rating_count'].quantile(0.8)
        movie_stats['popularity_score'] = (
            (movie_stats['rating_count'] / (movie_stats['rating_count'] + min_ratings)) *
            movie_stats['rating_avg'] +
            (min_ratings / (movie_stats['rating_count'] + min_ratings)) * 
            movie_stats['rating_avg'].mean()
        )
        
        popular_movies = movie_stats.nlargest(num_recommendations * 2, 'popularity_score')
        
        recommendations = []
        for _, row in popular_movies.iterrows():
            movie_match = self.movies[self.movies['movieId'] == row['movieId']]
            if not movie_match.empty:
                title = movie_match.iloc[0]['title']
                score = row['popularity_score']
                recommendations.append((title, score))
                if len(recommendations) >= num_recommendations:
                    break
        
        return recommendations
    
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
    with st.spinner('Loading...'):
        return MovieRecommender()

recommender = get_recommender()

st.title("Movie Recommender")

ai_mode = st.toggle("Use AI Recommendations", value=True, help="Uses machine learning collaborative filtering + content analysis")

if not ai_mode:
    st.info("Give recommendations based on genre filtering.")

similarity_threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.3, 0.05)

if st.button("Show Popular Movies"):
    popular = recommender.get_popular_recommendations(10)
    if popular:
        st.subheader("🏆 Most Popular Movies")
        for i, (title, score) in enumerate(popular, 1):
            st.write(f"{i}. **{title}** (Popularity Score: {score:.2f})")
    else:
        st.warning("Popular movies not available")

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
    movie_list = []
    for _, row in filtered_movies.iterrows():
        if pd.notna(row['year']):
            movie_list.append(f"{row['clean_title']} ({int(row['year'])})")
        else:
            movie_list.append(row['title'])
    
    selected_display = st.selectbox("Pick a movie:", sorted(movie_list))
    
    if selected_display and '(' in selected_display:
        clean_title = selected_display.split(' (')[0]
        matches = recommender.movies[recommender.movies['clean_title'] == clean_title]
        selected_movie = matches.iloc[0]['title'] if not matches.empty else selected_display
    else:
        selected_movie = selected_display

if selected_movie and st.button("Get Recommendations", type="primary"):
    st.session_state.recommendations_shown = 5
    st.session_state.current_movie = selected_movie
    st.session_state.similarity_threshold = similarity_threshold
    st.session_state.ai_mode = ai_mode
    
    if ai_mode:
        with st.spinner('Thinking about movie preferences...'):
            recommendations = recommender.get_recommendations(selected_movie, similarity_threshold, use_ai=True)
    else:
        recommendations = recommender.get_recommendations(selected_movie, similarity_threshold, use_ai=False)
    
    st.session_state.all_recommendations = recommendations or []

if st.session_state.get('all_recommendations'):
    mode_text = "AI Based" if st.session_state.get('ai_mode', True) else "Content-Based"
    st.subheader(f"Recommendations ({mode_text}):")
    
    for title, score in st.session_state.all_recommendations:
        if '(' in title and title.endswith(')'):
            st.write(f"🎬 **{title}** - *Similarity: {score:.3f}*")
        else:
            st.write(f"🎬 **{title}** - *Similarity: {score:.3f}*")

    if (st.session_state.get('current_movie') and 
        st.session_state.get('recommendations_shown', 5) < 100):
        
        if st.button("Show More"):
            ai_enabled = st.session_state.get('ai_mode', True)
            more_recs = recommender.get_extended_recommendations(
                st.session_state.current_movie,
                st.session_state.similarity_threshold,
                start_index=st.session_state.recommendations_shown,
                batch_size=5,
                use_ai=ai_enabled
            )
            
            if more_recs:
                st.session_state.all_recommendations.extend(more_recs)
                st.session_state.recommendations_shown += len(more_recs)
                st.rerun()
            else:
                st.info("No more recommendations available.")
                st.session_state.recommendations_shown = 100