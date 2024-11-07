import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Load model, vectorizer, and drug recommendations from pickle files
with open("model.pkl", "rb") as model_file, \
     open("tfidf.pkl", "rb") as vectorizer_file, \
     open("recommended_drug.pkl", "rb") as recommendations_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)
    drug_recommendations = pickle.load(recommendations_file)

# Streamlit App
st.title("Drug Recommendation System")
st.write("Enter a drug review to predict the condition and receive a drug recommendation.")

# Text input for the user review
user_review = st.text_area("Enter the drug review:")

if st.button("Predict Condition and Recommend Drug"):
    if user_review:
        # Preprocess the review text
        cleaned_review = preprocess_text(user_review)
        review_vector = vectorizer.transform([cleaned_review])  # Transform input using the loaded vectorizer
        
        # Predict condition using the loaded model
        predicted_condition = model.predict(review_vector)[0]
        
        # Recommend a drug based on the predicted condition
        recommended_drug = drug_recommendations.get(predicted_condition, "No recommendation available")
        
        # Display results
        st.write(f"**Predicted Condition:** {predicted_condition}")
        st.write(f"**Recommended Drug:** {recommended_drug}")
    else:
        st.write("Please enter a review to proceed.")
