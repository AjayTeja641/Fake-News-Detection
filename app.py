import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------- Custom CSS ----------
custom_css = """
<style>
/* Animated gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #1f1c2c, #928DAB, #434343, #000000);
    background-size: 600% 600%;
    animation: gradient 16s ease infinite;
    color: white;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Style for title and text */
h1, h2, h3, h4, h5, h6, .stTextInput>div>label, .stTextArea>div>label {
    color: #f1f1f1 !important;
}

/* Custom gradient button */
div.stButton > button {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    padding: 0.6em 2em;
    font-weight: bold;
    border: none;
    border-radius: 30px;
    transition: 0.3s ease-in-out;
}

div.stButton > button:hover {
    background: linear-gradient(to right, #4facfe, #00f2fe);
    color: black;
    transform: scale(1.05);
}

/* Sidebar customization */
[data-testid="stSidebar"] {
    background-color: #0f0f0f;
    color: white;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("🧠 Fake News Scanner")
st.sidebar.markdown("Created by **Ajay Teja K.**")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Fake_news.svg/512px-Fake_news.svg.png", width=150)
st.sidebar.markdown("🔗 [About Fake News](https://en.wikipedia.org/wiki/Fake_news)")

# ---------- Title ----------
st.title("📰 Fake News Detection with AI")

st.markdown("<p style='font-size:18px;'>Enter a news article to detect whether it's **REAL** or **FAKE** using a trained ML model.</p>", unsafe_allow_html=True)

# ---------- Input ----------
user_input = st.text_area("✍️ Paste or type the news content below:")

# ---------- Prediction ----------
if st.button("🔎 Predict Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter the news content.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        proba = model.predict_proba(transformed)[0]

        st.subheader("📊 Result")
        if prediction == 1:
            st.error("🚨 This news is likely **FAKE**.")
        else:
            st.success("✅ This news appears to be **REAL**.")

        # Confidence score
        confidence = round(np.max(proba) * 100, 2)
        st.info(f"🔍 Confidence Score: **{confidence}%**")
