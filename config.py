import os
import streamlit as st
from dotenv import load_dotenv

# Load local .env if it exists
load_dotenv()

def get_secret(key, default):
    """Retrieves secret from environment or streamlit secrets."""
    # 1. Try Environment Variable (Local or OS-level)
    val = os.getenv(key)
    if val and val != default:
        return val
    
    # 2. Try Streamlit Secrets (Cloud Deployment)
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
        
    return default

# Gemini API Configuration - MANDATORY for Embeddings
GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

# Groq API Configuration - OPTIONAL for Reasoning
GROQ_API_KEY = get_secret("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

# Model Configuration
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
GENERATIVE_MODEL_GEMINI = "gemini-1.5-flash"
GENERATIVE_MODEL_GROQ = "llama-3.3-70b-versatile"

# FAISS Configuration
FAISS_INDEX_PATH = "faiss_index"

# Extraction Configuration
TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Application Flags
USE_GROQ = True if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY" else False
