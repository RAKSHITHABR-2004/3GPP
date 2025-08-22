3GPP Release Comparison & Query Assistant

This project is an AI-powered assistant for 3GPP documentation. It provides two main functionalities:

» Document Embedding & Storage: Converts 3GPP documents into vector embeddings for semantic search and retrieval.
» Version Comparison: Detects differences between two 3GPP releases using embeddings and similarity analysis.
» Streamlit Chat Interface: Allows users to interact with the system via a web-based chatbot UI with login/signup support.

Features

» Upload and Embed Documents – Converts 3GPP PDFs/DOCX/HTML into embeddings and stores them in ChromaDB.
» Semantic Search – Retrieves relevant chunks based on user queries using HuggingFace sentence-transformers.
» Version Comparison – Compares two releases (e.g., Release 12 vs Release 13) and outputs modified, added, and deleted sections.
» Change Detection – Uses cosine similarity + Jaccard similarity for semantic difference classification.
» Persistent Storage – All embeddings and chat history are stored in ChromaDB for fast retrieval.
» Interactive Chat UI – Built with Streamlit, includes user authentication (login/signup) and chat history management.
» LLM Integration – Uses Cohere API for natural language processing (summary & comparison).

Project Structure
├── embed2.py      # Handles document loading, text chunking, embedding, and storing in ChromaDB
├── release8.py    # Streamlit-based chat interface for querying and interacting with the assistant
├── wver2.py       # Handles version comparison, semantic change detection, and summary embedding
└── chroma_storage # Directory for persistent ChromaDB storage (auto-created)


Tech Stack

• Language: Python 3.x
• Libraries:
    Streamlit – UI framework
    Cohere – LLM for summaries & comparisons
    HuggingFace Sentence Transformers – Embeddings
    ChromaDB – Vector database for storage & retrieval
    scikit-learn – Cosine similarity
    LlamaIndex – Document loading
• Database: ChromaDB (persistent local storage)


How It Works
1. Generate Embeddings
Run the embed2.py script to upload a document (PDF, DOCX, or HTML), split it into chunks, create embeddings, and store them in ChromaDB:
python embed2.py

2. Compare Two Releases
Run wver2.py to compare two releases. It checks for missing embeddings, computes cosine similarity, and classifies changes:
export VERSION1=re12
export VERSION2=re13
python wver2.py
Output: change_detection_re12_vs_re13.json with detailed added/removed/modified sections.

3. Launch Streamlit App
Start the chat interface for querying and comparing:
streamlit run release8.py

Features:
User Login/Signup
Chat-based Q&A about 3GPP
Summarize a release or compare two releases
Displays retrieved context and LLM-generated responses

Environment Variables:
CHROMA_PATH – Path to ChromaDB storage (default: ../chroma_storage)
VERSION1, VERSION2 – Release names for comparison
COHERE_API_KEY – Your Cohere API key for LLM responses

Outputs:
Embeddings JSON: <release_name>_embeddings.json
Comparison Report: change_detection_<version1>_vs_<version2>.json

ChromaDB Collections:
One collection per release (e.g., re12, re13)
One collection for comparisons (e.g., re12_vs_re13)

Future Enhancements:
Add support for more embedding models (e.g., OpenAI, local LLMs)
Enhance UI with file upload & progress tracking
Add fine-grained change visualization in Streamlit


Quick Demo Workflow

1.Run embed2.py for Release 10 & Release 17.
2.Run wver2.py for comparison.
3.Start release8.py and ask:

Compare release 10 and release 17
is MUSIM feature supported in rel17?
Is PS SRVCC handover a new feature in Rel 17 ?
What are the new features in release 17?
Summarize release 10