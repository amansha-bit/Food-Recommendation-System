import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib  # To save and load the collaborative filtering model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import PorterStemmer
import re

# Load data
df = pd.read_csv('dataset.csv')  # Replace with the actual path

# Assuming that 'rating' is your original rating dataset
rating = pd.read_csv('ratings.csv')
rating_matrix = rating.pivot_table(index='Food_ID', columns='User_ID', values='Rating').fillna(0)
rating_matrix.to_csv('rating_matrix.csv', index=True)

# Assuming 'rating_matrix' is the correct matrix with consistent features
csr_rating_matrix = csr_matrix(rating_matrix.values)
recommender = NearestNeighbors(metric='cosine')
recommender.fit(csr_rating_matrix)

# Clean up the 'Veg_Non' column by removing leading and trailing spaces
df['Veg_Non'] = df['Veg_Non'].str.strip()

# Create a Porter Stemmer
stemmer = PorterStemmer()

# Function to stem words and remove numeric characters
def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())  # Tokenize and convert to lowercase
    words = [word for word in words if not any(char.isdigit() for char in word)]  # Remove numeric words
    stemmed_words = [stemmer.stem(word) for word in words]  # Stemming
    return stemmed_words

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=preprocess_text)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Describe'].fillna(''))

# Sidebar
st.sidebar.title("Food Recommendations")

# Dropdown for recommendation type
recommendation_type = st.sidebar.selectbox("Select Recommendation Type", ["Detailed Recommendation", "Ingredient Based Recommendation"])

if recommendation_type == "Detailed Recommendation":
    # Detailed Recommendation

    # Select Veg/Non-Veg
    veg_non_veg = st.sidebar.radio("Select Veg/Non-Veg", ["All", "Veg", "Non-Veg"])

    # Filter food types based on Veg/Non-Veg selection
    if veg_non_veg == "All":
        filtered_food_types = df['C_Type'].unique()
    else:
        filtered_food_types = df[df['Veg_Non'].str.lower() == veg_non_veg.lower()]['C_Type'].unique()

    # Select Type of Food
    selected_food_type = st.sidebar.selectbox("Select Type of Food", filtered_food_types)

    # Filter food names based on Veg/Non-Veg and selected food type
    if veg_non_veg == "All":
        filtered_food_names = df[df['C_Type'] == selected_food_type]['Name'].unique()
    else:
        filtered_food_names = df[(df['Veg_Non'].str.lower() == veg_non_veg.lower()) & (df['C_Type'] == selected_food_type)]['Name'].unique()

    # Select Food Name
    selected_food_name = st.sidebar.selectbox("Select Food Name", filtered_food_names)

    # Recommendations
    if st.sidebar.button("Get Recommendations"):
        if selected_food_name:
            user_data = df[df['Name'] == selected_food_name]

            if not user_data.empty:
                user_index = rating_matrix.index[rating_matrix.index == int(user_data['Food_ID'].values[0])]

                if not user_index.empty:
                    user_index = user_index[0]
                    user_ratings = rating_matrix.loc[user_index]

                    reshaped = user_ratings.values.reshape(1, -1)
                    distances, indices = recommender.kneighbors(reshaped, n_neighbors=16)

                    nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]
                    nearest_neighbors = pd.DataFrame({'Food_ID': nearest_neighbors_indices})

                    # Filter recommendations based on Veg/Non-Veg and selected food type
                    filtered_recommendations = df[df['Food_ID'].isin(nearest_neighbors_indices)]
                    filtered_recommendations = filtered_recommendations[filtered_recommendations['C_Type'] == selected_food_type]
                    filtered_recommendations = filtered_recommendations[filtered_recommendations['Veg_Non'].str.lower() == veg_non_veg.lower()]

                    st.write(f"Recommended Food Items for {selected_food_name}:")
                    st.table(filtered_recommendations[['Name', 'C_Type', 'Veg_Non']])
                else:
                    st.warning("Food not found in the rating matrix.")
            else:
                st.warning("Food not found in the dataset.")
        else:
            st.warning("Please select a food name.")

elif recommendation_type == "Ingredient Based Recommendation":
    # Ingredient Based Recommendation

    # Multiselect for Ingredients
    all_ingredients = tfidf_vectorizer.get_feature_names_out()
    unique_stemmed_ingredients = set(preprocess_text(' '.join(all_ingredients)))

    selected_ingredients = st.sidebar.multiselect("Select Ingredients", list(unique_stemmed_ingredients))

    # Button to get recommendations
    if st.sidebar.button("Get Food Recommendations"):
        if selected_ingredients:
            # Create a TF-IDF vector for selected ingredients
            selected_tfidf_matrix = tfidf_vectorizer.transform([' '.join(selected_ingredients)])

            # Calculate cosine similarity between selected ingredients and food descriptions
            cosine_similarities = linear_kernel(selected_tfidf_matrix, tfidf_matrix).flatten()

            # Get top 5 most similar food items
            top_indices = cosine_similarities.argsort()[:-6:-1]
            recommended_foods = df.iloc[top_indices]

            # Display recommendations
            st.write("Recommended Foods:")
            st.table(recommended_foods[['Name', 'C_Type', 'Veg_Non']])
        else:
            st.warning("Please select at least one ingredient.")
