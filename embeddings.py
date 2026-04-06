import os
import numpy as np
import faiss
import pickle
from google import genai
from google.genai import types
from PIL import Image
import config
from typing import List, Dict, Any
import tenacity

# Initialize Gemini Client
def get_client():
    return genai.Client(api_key=config.GEMINI_API_KEY)

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception)
)
def _embed_batch(client, model, batch):
    return client.models.embed_content(
        model=model,
        contents=batch
    )

def get_text_embeddings(texts: List[str]) -> np.ndarray:
    """Generates embeddings for a list of text chunks with retries."""
    if not texts:
        return np.array([]).reshape(0, 0)
        
    client = get_client()
    embeddings = []
    try:
        # Gemini 2 might have limits on batch size
        for i in range(0, len(texts), 20):
            batch = texts[i:i+20]
            response = _embed_batch(client, config.EMBEDDING_MODEL, batch)
            for emb in response.embeddings:
                embeddings.append(emb.values)
    except Exception as e:
        print(f"Failed to generate text embeddings after retries: {e}")
        # Return zeros as fallback or raise? Let's return empty to handle in UI
        return np.array([]).reshape(0, 0)
        
    return np.array(embeddings).astype('float32')

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10)
)
def get_image_embedding(image_path: str) -> np.ndarray:
    """Generates an embedding for an image file using Gemini 2 with retries."""
    if not os.path.exists(image_path):
        return np.zeros(768).astype('float32') # Fallback dimension
        
    client = get_client()
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Multimodal embedding request
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ]
        )
        return np.array(response.embeddings[0].values).astype('float32')
    except Exception as e:
        print(f"Failed to generate image embedding for {image_path}: {e}")
        return np.zeros(768).astype('float32')

def get_table_embeddings(tables: List[Dict[str, Any]]) -> np.ndarray:
    """Generates embeddings for table data by converting to string first."""
    table_strings = []
    for t in tables:
        # Convert table (list of lists) to a string representation
        table_str = "\n".join([" | ".join([str(cell) for cell in row]) for row in t["table"]])
        table_strings.append(f"Table on page {t['page']}:\n{table_str}")
    
    return get_text_embeddings(table_strings)

class VectorDB:
    def __init__(self, dimension: int = 768): # Gemini 2 default might be different, let's check or handle dynamically
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_vectors(self, vectors: np.ndarray, meta_list: List[Dict[str, Any]]):
        if vectors.shape[1] != self.dimension:
            # Re-initialize index if dimension mismatch occurs first time
            if self.index.ntotal == 0:
                self.dimension = vectors.shape[1]
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"Dimension mismatch: {vectors.shape[1]} vs {self.dimension}")
        
        self.index.add(vectors)
        self.metadata.extend(meta_list)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                meta = self.metadata[idx].copy()
                meta["score"] = float(dist)
                results.append(meta)
        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str):
        index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", "rb") as f:
            metadata = pickle.load(f)
        db = cls(dimension=index.d)
        db.index = index
        db.metadata = metadata
        return db

if __name__ == "__main__":
    # Test Embeddings (requires real API key)
    # vdb = VectorDB()
    pass
