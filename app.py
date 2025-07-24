import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detection App")

input_text = st.text_area("Enter the news content to check:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some news content.")
    else:
        # Preprocess and predict
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]

        if prediction == 1:
            st.error("⚠️ This news is likely **FAKE**.")
        else:
            st.success("✅ This news appears to be **REAL**.")
