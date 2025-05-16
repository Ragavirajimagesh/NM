mport streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("best_model_nb.joblib")
vectorizer = joblib.load("best_tfidf.joblib")

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text.lower()

def preprocess(text):
    words = clean_text(text).split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("Social Media Sentiment Classifier")
user_input = st.text_area("Enter your social media post:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)
        st.success(f"Predicted Sentiment: {prediction[0]}")
    else:
        st.warning("Please enter some text.")
