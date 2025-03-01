#Document Search Using Ollama and Deepseek R1

Overview
This is a Streamlit-based application that allows users to upload article URLs, process the content, and ask questions about the extracted text using Deepseek R1 via Ollama.

Features
Extracts text from up to 3 user-provided URLs.
Chunks the extracted text for efficient retrieval.
Computes embeddings using SentenceTransformers (all-MiniLM-L6-v2).
Stores processed text and embeddings for faster querying.
Uses cosine similarity to retrieve the most relevant text chunks.
Sends a context-aware query to Deepseek R1 via Ollama for answer generation.
Displays sources used for answering questions.

Requirements:
System Requirements
Python 3.8+
Streamlit
Ollama (installed and running locally)
Deepseek R1 model pulled in Ollama

Python Dependencies
Install the required dependencies:

sh
pip install streamlit requests beautifulsoup4 sentence-transformers scikit-learn numpy pickle5

Usage
1. Start Ollama
Make sure Ollama is running locally:

sh
ollama serve

2. Pull the Deepseek R1 model
Ensure you have downloaded the Deepseek R1 model for Ollama:

sh
ollama pull deepseek-r1:1.5b

3. Run the Streamlit App
Start the Streamlit application by running:

sh
streamlit run app.py

How to Use
Enter URLs: Provide up to 3 article URLs in the sidebar.
Process URLs: Click the "Process URLs" button to extract, chunk, and embed the text.
Ask Questions: Type a question related to the articles
View Results: See the AI-generated answer along with cited sources.

Troubleshooting
If you encounter issues:

Check if Ollama is running: Run ollama serve in your terminal.

Ensure the model is pulled: Run ollama pull deepseek-r1:1.5b.

Verify URL processing: Ensure the URLs provided contain valid textual content.

Future Enhancements
Support for additional language models.

Improved chunking strategies.

Caching for faster response times.

Support for PDF and text document uploads.

License
This project is licensed under the MIT License.
