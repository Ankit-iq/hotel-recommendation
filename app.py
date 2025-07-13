from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load saved data
with open('sentence_embeddings.pkl', 'rb') as f:
    sentence_embeddings = pickle.load(f)

df = pd.read_pickle('hotel_data_with_predictions.pkl')

# Load Sentence-BERT model
model = SentenceTransformer('all-mpnet-base-v2')

# Recommendation function
def recommend_hotels(input_text, top_n=5):
    input_embedding = model.encode([input_text])

    # Combine all cities and states into a set of lowercase names
    all_locations = set(df['city'].str.lower()) | set(df['state'].str.lower())

    # Extract matching cities/states from user input
    matching_locations = [loc for loc in all_locations if loc in input_text.lower()]

    if matching_locations:
        # Filter dataframe based on matched city/state
        filtered_df = df[df['city'].str.lower().isin(matching_locations) | df['state'].str.lower().isin(matching_locations)]

        if not filtered_df.empty:
            # Recompute sentence embeddings for filtered hotels
            hotel_sentences = (filtered_df['property_name'] + ' in ' + filtered_df['city'] + ', ' + filtered_df['state']).tolist()
            filtered_embeddings = model.encode(hotel_sentences)
            similarities = cosine_similarity(input_embedding, filtered_embeddings)[0]
            top_indices = similarities.argsort()[-top_n:][::-1]
            recommendations = filtered_df.iloc[top_indices][['property_name', 'city', 'state', 'site_review_rating']].copy()
            recommendations['similarity'] = similarities[top_indices]
            return recommendations

    # Fallback: If no matching location or filtered list is empty, use original full dataset
    similarities = cosine_similarity(input_embedding, sentence_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices][['property_name', 'city', 'state', 'site_review_rating']].copy()
    recommendations['similarity'] = similarities[top_indices]
    return recommendations


# Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    if request.method == 'POST':
        user_input = request.form['query']
        recommendations = recommend_hotels(user_input).to_dict(orient='records')
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True,port=5000)
