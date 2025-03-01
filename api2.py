import streamlit as st
import requests
import json
import numpy as np
import pickle
import time
import os
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Set page title
st.set_page_config(page_title="Document Search", layout="wide")

# Main UI elements
st.title("Document Search")
st.write("Upload article URLs and ask questions about them!")

# Sidebar for URL inputs
st.sidebar.title("Article/Documents URLs")
st.sidebar.write("Enter up to 3 URLs to articles you want to analyze")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url:  # Only add non-empty URLs
        urls.append(url)

# File path for storing the processed data
data_file_path = "processed_articles.pkl"

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Clean text: break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text, url
    except Exception as e:
        st.error(f"Error extracting text from {url}: {str(e)}")
        return None, url

# Function to chunk text
def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and end - start >= chunk_size:
            # Find the last period or newline to avoid cutting sentences
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
            elif last_newline > start + chunk_size // 2:
                end = last_newline + 1
        
        chunks.append({
            "text": text[start:end],
            "start": start,
            "end": end
        })
        
        # Move start position for next chunk, with overlap
        start = end - overlap if end < text_length else text_length
    
    return chunks

# Main area for processing and results
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # Process URLs button
    process_url_clicked = st.button("Process URLs", disabled=len(urls) == 0)
    
    status_placeholder = st.empty()
    
    if process_url_clicked and urls:
        try:
            # Initialize sentence transformer model for embeddings
            status_placeholder.info("Loading embedding model...")
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Process each URL
            status_placeholder.info("Extracting text from URLs...")
            all_chunks = []
            article_sources = []
            
            for url in urls:
                text, source = extract_text_from_url(url)
                if text:
                    chunks = chunk_text(text)
                    for chunk in chunks:
                        chunk["source"] = source
                    all_chunks.extend(chunks)
                    article_sources.append(source)
            
            if not all_chunks:
                status_placeholder.error("No text could be extracted from the provided URLs.")
                st.stop()
                
            # Create embeddings for chunks
            status_placeholder.info("Creating embeddings for text chunks...")
            chunk_texts = [chunk["text"] for chunk in all_chunks]
            chunk_embeddings = embedding_model.encode(chunk_texts)
            
            # Save processed data
            processed_data = {
                "chunks": all_chunks,
                "embeddings": chunk_embeddings,
                "sources": article_sources
            }
            
            with open(data_file_path, "wb") as f:
                pickle.dump(processed_data, f)
                
            status_placeholder.success("✅ Processing complete! You can now ask questions about the articles.")
            
            # Store the article sources in session state
            st.session_state['article_sources'] = article_sources
            
        except Exception as e:
            status_placeholder.error(f"Error processing URLs: {str(e)}")

    # Display loaded article sources
    if 'article_sources' in st.session_state and st.session_state['article_sources']:
        st.subheader("Loaded Articles:")
        for source in st.session_state['article_sources']:
            st.write(f"- {source}")

with main_col2:
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Select Ollama Model",
        ["deepseek-r1:1.5b"],  # Updated to use the new model
        help="Choose the Deepseek model version to use with Ollama"
    )
    
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="How many text chunks to use for answering questions"
    )

# Function to query Ollama
def query_ollama(prompt, model_name, temperature=0.7):
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,  # Use the model name selected by the user
                'prompt': prompt,
                'stream': False,
                'temperature': temperature
            }
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# Query input section
st.divider()
st.subheader("Ask Questions About Your Articles")
query = st.text_input("Your question:", placeholder="What are the main points discussed in these articles?")

# Query processing
if query and os.path.exists(data_file_path):
    try:
        # Setup progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load processed data
        status_text.text("Loading knowledge base...")
        progress_bar.progress(20)
        
        with open(data_file_path, "rb") as f:
            processed_data = pickle.load(f)
            
        chunks = processed_data["chunks"]
        chunk_embeddings = processed_data["embeddings"]
        
        # Get embedding for the query
        status_text.text("Processing your question...")
        progress_bar.progress(40)
        
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = embedding_model.encode([query])[0]
        
        # Calculate similarity and get top chunks
        progress_bar.progress(60)
        
        # Calculate cosine similarity between query and all chunks
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Get indices of top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Get the top chunks
        top_chunks = [chunks[i] for i in top_indices]
        
        # Prepare context from top chunks
        context = "\n\n".join([f"CHUNK FROM {chunk['source']}:\n{chunk['text']}" for chunk in top_chunks])
        
        # Prepare prompt for Ollama
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        
CONTEXT:
{context}

QUESTION:
{query}

Please answer the question based only on the provided context. If you don't know the answer from the context, say so.
Include sources for your information by mentioning which article URLs contain the relevant information.
"""
        
        # Query Ollama
        status_text.text("Generating answer with Deepseek R1...")
        progress_bar.progress(80)
        
        answer = query_ollama(prompt, model_name, temperature)
        
        progress_bar.progress(100)
        status_text.empty()
        
        # Display results
        st.header("Answer")
        st.write(answer)
        
        # Display sources used
        st.subheader("Sources Used:")
        source_set = set()
        for chunk in top_chunks:
            source_set.add(chunk["source"])
        
        for source in source_set:
            st.write(f"- {source}")
    
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        st.info("Tips for troubleshooting:\n"
                "1. Make sure Ollama is running locally on your machine\n"
                "2. Check that you've pulled the Deepseek model with: ollama pull deepseek-r1:1.5b\n"
                "3. Ensure your URLs were processed successfully")
else:
    if not os.path.exists(data_file_path) and query:
        st.warning("Please process some URLs first before asking questions!")

# Instructions at the bottom
with st.expander("📋 How to use this tool"):
    st.markdown(""" 
    1. *Add URLs*: Enter up to 3 news article URLs in the sidebar
    2. *Process URLs*: Click the 'Process URLs' button to load and analyze the articles
    3. *Ask Questions*: Type your question in the text box and press Enter
    4. *View Results*: See the answer generated by Deepseek R1 with relevant sources
    
    *Requirements*:
    - This tool requires [Ollama](https://ollama.ai/) running on your machine
    - Make sure you've pulled the Deepseek model: ollama pull deepseek-r1:1.5b
    - Ollama should be accessible at http://localhost:11434
    """)
