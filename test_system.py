import os
import sys
import config
from google import genai
import faiss
import numpy as np
from groq import Groq

def test_gemini():
    print("Checking Gemini API...")
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print("ERROR: Gemini API Key is missing or default. Add it to config.py or .env.")
        return False
    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents="test"
        )
        if hasattr(response, 'embeddings'):
            print("SUCCESS: Gemini API is working correctly.")
            return True
    except Exception as e:
        print(f"ERROR: Gemini API Error: {e}")
        return False

def test_groq():
    print("Checking Groq API...")
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        print("INFO: Groq API Key is not set. System will default to Gemini for reasoning.")
        return True
    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model=config.GENERATIVE_MODEL_GROQ,
        )
        if chat_completion.choices:
            print("SUCCESS: Groq API is working correctly.")
            return True
    except Exception as e:
        print(f"ERROR: Groq API Error: {e}")
        return False

def test_faiss():
    print("Checking FAISS...")
    try:
        index = faiss.IndexFlatL2(768)
        data = np.random.random((5, 768)).astype('float32')
        index.add(data)
        if index.ntotal == 5:
            print("SUCCESS: FAISS is working correctly.")
            return True
    except Exception as e:
        print(f"ERROR: FAISS Error: {e}")
        return False

if __name__ == "__main__":
    print("=== Multimodal Supply Chain Intelligence Health Check ===\n")
    g_ok = test_gemini()
    gr_ok = test_groq()
    f_ok = test_faiss()
    
    if g_ok and gr_ok and f_ok:
        print("\nSUCCESS: All systems are GO! You can now run 'streamlit run app.py'.")
    else:
        print("\nWARNING: Some mandatory systems are failing. Please check your config.py or environment variables.")
