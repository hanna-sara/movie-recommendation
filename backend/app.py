from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# TMDB API Key (use your real key in a .env file or environment variable)
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "PLACEHOLDER_KEY")  # Replace with your key later

# Load and preprocess movie data
df = pd.read_csv("movie_dataset.csv")
df = df.dropna(subset=["genres", "director"])
df["combined_features"] = df["genres"] + " " + df["director"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(df.index, index=df["original_title"]).drop_duplicates()

# Function to get TMDB poster URL
def get_movie_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print("TMDB API error:", e)
    return None

# Movie recommendation logic
def recommend_movies(title, num_recommendations=5):
    if title not in movie_indices:
        return []
    index = movie_indices[title]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:num_recommendations + 1]
    
    recommended = []
    for i in similarity_scores:
        movie_title = df["original_title"].iloc[i[0]]
        poster_url = get_movie_poster(movie_title)
        recommended.append({
            "title": movie_title,
            "poster": poster_url
        })
    return recommended

# API endpoint for recommendations
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    title = data.get("movie_name", "")
    results = recommend_movies(title)
    return jsonify({"recommendations": results})

if __name__ == "__main__":
    app.run(debug=True)
