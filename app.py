import streamlit as st
import pickle
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- Load NLTK data ---
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# --- Initialize ---
word_lema = WordNetLemmatizer()
stop_word = set(stopwords.words("english"))

# --- Load Model and Vectorizer ---
model = pickle.load(open("model.pkl", "rb"))
vector = pickle.load(open("vec.pkl", "rb"))

# --- Text normalization function ---
def normalize(text):
    text = text.lower()
    text = re.sub(r'(.)\1+', r'\1', text)  # remove repeated letters
    text = re.sub(r'[^\w\s]', "", text)  # remove punctuation
    text = re.sub(r"\d", "", text)  # remove digits
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"isn't", "isnot", text)
    text = re.sub(r'http[s]?://\S+', "", text)  # remove URLs
    token = word_tokenize(text)
    text = [word_lema.lemmatize(word) for word in token if word not in stop_word]
    return " ".join(text)

# --- Streamlit UI ---
st.title("📩 Spam Message Detector")
st.write("Enter a message below and check if it's **Spam or Not Spam (Ham)**")

# Text input
user_input = st.text_area("Type your message here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        # Normalize and vectorize
        clean_text = normalize(user_input)
        transformed_text = vector.transform([clean_text])

        # Prediction
        prediction = model.predict(transformed_text)[0]

        # Show result
        if prediction == 1:  # Assuming 1 = Spam
            st.error("🚨 This message is **SPAM!**")
        else:
            st.success("✅ This message looks **SAFE (Not Spam)**.")
