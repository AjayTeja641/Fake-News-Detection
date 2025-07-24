import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------- Custom Styles ----------
custom_css = """
<style>
/* Background animation using gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #1e1e1e, #2e2e2e, #3e3e3e, #1e1e1e);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

[data-testid="stSidebar"] {
    background-color: #111;
    color: white;
}

h1, h2, h3, h4, h5, h6 {
    color: #f8f8f8 !important;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

footer, header {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Fake_news.svg/512px-Fake_news.svg.png", width=150)
st.sidebar.title("🔧 Settings")
dark_mode = st.sidebar.toggle("🌙 Enable Dark Mode", value=True)
st.sidebar.markdown("Developed by **Ajay Teja K.**")
st.sidebar.markdown("---")
st.sidebar.markdown("📘 [Learn about Fake News](https://en.wikipedia.org/wiki/Fake_news)")

# ---------- Page Title ----------
st.title("📰 AI-Powered Fake News Detector")
st.markdown(
    "<p style='font-size:18px;'>Enter a news article to check whether it's real or fake using Machine Learning.</p>",
    unsafe_allow_html=True
)

# ---------- Text Input ----------
input_text = st.text_area("✍️ Paste or type the news content here:", height=200)

# ---------- Prediction Logic ----------
if st.button("🔎 Predict Now"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]
        proba = model.predict_proba(transformed_text)[0]

        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.error("🚨 The news is likely **FAKE**.")
        else:
            st.success("✅ The news appears to be **REAL**.")

        # Confidence score
        confidence = round(np.max(proba) * 100, 2)
        st.info(f"🔍 Confidence Score: **{confidence}%**")

        # Pie Chart (optional)
        st.markdown("### 📈 Probability Breakdown")
        st.progress(confidence / 100)
