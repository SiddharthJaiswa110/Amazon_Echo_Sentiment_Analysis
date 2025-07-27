# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pickle
import pandas as pd

def predict():
    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    
st.set_page_config(
    page_title="Amazon Echo Sentiment Analysis",
    layout="wide",
    page_icon="🔊"
)

# Sidebar for navigation
with st.sidebar:
    selected = st.radio(
        'Menu',
        ['Sentiment Prediction'],
        index=0
    )

# Main Page
if selected == 'Sentiment Prediction':
    st.title("Amazon Echo Review Sentiment Analysis")

    input_type = st.radio(
        "Choose input method:",
        ("Type review", "Upload CSV file")
    )

    if input_type == "Type review":
        review_text = st.text_area("Enter your Amazon Echo review below:")

        if st.button("Analyze Sentiment"):
            if review_text:
                review_vec = vectorizer.transform([review_text])
                pred = model.predict(review_vec)[0]
                st.success(f"Predicted Sentiment: *{pred}*")
            else:
                st.warning("Please enter some review text.")

    else:  # File upload
        uploaded = st.file_uploader("Upload a CSV file containing a 'review' column", type="csv")
        if uploaded and st.button("Analyze File"):
            df = pd.read_csv(uploaded)
            if 'review' not in df.columns:
                st.error("CSV must have a column named 'review'")
            else:
                vecs = vectorizer.transform(df['review'].astype(str))
                preds = model.predict(vecs)
                df['Predicted Sentiment'] = preds
                st.dataframe(df[['review', 'Predicted Sentiment']])