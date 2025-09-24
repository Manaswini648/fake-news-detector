import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Paste a news article or tweet. The AI will classify it as *Real*, *Fake*, or *Misleading*.")

user_input = st.text_area("Enter news text below:", height=200)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing using Hugging Face API..."):
            labels = ["real", "fake", "misleading"]
            result = query({
                "inputs": user_input,
                "parameters": {"candidate_labels": labels}
            })

            st.subheader("🔍 Prediction Results")
            try:
                for label, score in zip(result["labels"], result["scores"]):
                    st.write(f"**{label.capitalize()}**: {round(score * 100, 2)}%")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
