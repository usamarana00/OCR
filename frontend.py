import streamlit as st
import requests
from dotenv import load_dotenv
import os
load_dotenv()

url = os.getenv("URL")

st.title("Local OCR Application")

# Let the user upload an image file
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Perform OCR"):
        # Prepare the file data for the API request
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            result = response.json()
            st.subheader("OCR Result")
            st.write("**Raw Document:**", result["raw_document"])
            st.write("**Markdown Format:**", result["markdown"])
        except requests.RequestException as e:
            st.error(f"OCR request failed: {e}")
