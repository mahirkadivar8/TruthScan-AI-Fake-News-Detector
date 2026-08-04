import streamlit as st
import pickle

# Load model and vectorizer
with open('model/fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# App title
st.title("TruthScan - AI Fake News Detector")
st.write("Enter a news article to check if it is Real or Fake!")

# Text input
news_text = st.text_area("Paste news article here:", height=200)

# Predict button
if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Vectorize input
        input_tfidf = tfidf.transform([news_text])
        # Predict
        prediction = model.predict(input_tfidf)[0]
        confidence = model.predict_proba(input_tfidf)[0]
        
        if prediction == 0:
            st.error(f"FAKE NEWS! Confidence: {confidence[0]*100:.2f}%")
        else:
            st.success(f"REAL NEWS! Confidence: {confidence[1]*100:.2f}%")