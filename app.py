import streamlit as st
import pandas as pd
import joblib
import re

model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

st.title("📧 Spam Email Classifier")
st.write("Enter an email text below and the model will classify it as Spam or Not Spam.")

email_text = st.text_area("Email content:", height=200)

if st.button("Classify"):
    if email_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        cleaned = clean_text(email_text)
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        confidence = max(prob)

        if prediction == "spam":
            st.error(f"🚨 This email is SPAM (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ This email is NOT SPAM (Confidence: {confidence:.2f})")

st.sidebar.header("Dataset Preview")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write(df.head())