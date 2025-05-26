import streamlit as st
import tempfile
import os
import sys

# Add parent directory to sys.path to find src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils import TranscriptLoader
from src.search_tfidf import TFIDFSearcher
from src.search_hf import HuggingFaceSearcher

st.set_page_config(page_title="Semantic Search for Transcript Q&A", layout="wide")

st.title("Semantic Search for Transcript Q&A")
st.markdown("""
Upload a transcript file, select a search method, and ask a question to find the most relevant chunk.
The transcript should be in the format `[MM:SS - MM:SS] text`.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Transcript File", type=["txt"])

if uploaded_file:
    # Save transcript to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
    
    # Load transcript
    loader = TranscriptLoader(temp_path)
    chunks, timestamps = loader.load_and_chunk()
    os.unlink(temp_path)
    
    if not chunks:
        st.error("Failed to load transcript. Please ensure the file is in the correct format.")
        st.stop()

    # Search method selection
    method = st.selectbox("Search Method", ["TF-IDF", "Hugging Face LLM (llm2)"])

    # Initialize searcher
    searcher = None
    if method == "TF-IDF":
        searcher = TFIDFSearcher(chunks, timestamps)
    elif method == "Hugging Face LLM (llm2)":
        searcher = HuggingFaceSearcher(chunks, timestamps)

    if searcher is None:
        st.error("Failed to initialize search method.")
        st.stop()

    # Query input
    question = st.text_input("Enter your question:", placeholder="e.g., What is artificial intelligence?")

    if st.button("Search"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching..."):
                output_file_path = 'output/output.txt'
                try:
                    result = searcher.search(question)
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    with open(output_file_path, 'a', encoding='utf-8') as out_file:
                        if result:
                            timestamp, text_chunk = result
                            result_display = f"[{timestamp}], {text_chunk}"
                            st.success(f"**Result:** {result_display}")
                            log_output = f"Question: {question}\nMethod: {method.lower()}\nOutput: {result_display}\n\n"
                        else:
                            st.warning("No relevant answer found.")
                            log_output = f"Question: {question}\nMethod: {method.lower()}\nOutput: No relevant answer found.\n\n"
                        out_file.write(log_output)
                except Exception as e:
                    st.error(f"Error during search: {e}")
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    with open(output_file_path, 'a', encoding='utf-8') as out_file:
                        log_output = f"Question: {question}\nMethod: {method.lower()}\nOutput: Error - {e}\n\n"
                        out_file.write(log_output)
else:
    st.info("Please upload a transcript file to begin.")

st.markdown("---")
st.markdown("Built with Streamlit for the AI Internship Assignment, May 2025.")