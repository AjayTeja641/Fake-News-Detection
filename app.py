import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Background image using custom CSS
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1590650046871-92c887180603?fit=crop&w=1650&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    backdrop-filter: blur(4px);
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
.title {
    color: #fff;
    text-align: center;
    padding-top: 20px;
}
.stTextArea label, .stButton button {
    color: #fff !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="title">📰 Fake News Detection Portal</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size:18px; color:white;'>Enter news content below and check if it’s real or fake using our AI model!</p>",
    unsafe_allow_html=True,
)

# Input text area
input_text = st.text_area("🖊️ Enter the news content to check:", height=200)

if st.button("🚨 Predict"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]

        if prediction == 1:
            st.error("❌ This news is likely **FAKE**.")
        else:
            st.success("✅ This news appears to be **REAL**.")
