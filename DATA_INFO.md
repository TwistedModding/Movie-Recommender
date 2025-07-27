# Dataset Information

This movie recommender uses the MovieLens dataset, which is automatically downloaded when you first run the application.

## What happens when you run the app:

1. **First time**: The app will automatically download the MovieLens dataset (ml-latest-small) from the official GroupLens website
2. **Subsequent runs**: The app will use the cached data from your local `data/` folder

## Dataset Details:

- **Source**: [MovieLens](https://grouplens.org/datasets/movielens/) by GroupLens Research
- **Size**: ~1MB (small dataset with ~9,000 movies)
- **License**: Free to use for research and development
- **Content**: Movie titles, genres, ratings, and tags

## Manual Download (Optional):

If you prefer to download the dataset manually:

1. Visit: https://grouplens.org/datasets/movielens/
2. Download "ml-latest-small.zip"
3. Extract the contents to the `data/` folder in this project
4. The app will automatically detect and use your local data

## Privacy:

- The data folder is excluded from version control (see `.gitignore`)
- No movie data is stored in the repository
- Dataset is downloaded directly from the official source
