import numpy as np
import config
from embeddings import VectorDB, get_text_embeddings
from google import genai
from groq import Groq
from typing import List, Dict, Any
import tenacity

# Define Gemini Client
gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

# Define Groq Client (if key exists)
groq_client = None
if config.GROQ_API_KEY and config.GROQ_API_KEY != "YOUR_GROQ_API_KEY":
    groq_client = Groq(api_key=config.GROQ_API_KEY)

def retrieve_from_vector_db(query: str, vdb: VectorDB, k: int = 5) -> List[Dict[str, Any]]:
    """Retrieves relevant text, tables, and images from FAISS using Gemini 2."""
    query_vector = get_text_embeddings([query])[0]
    if query_vector.size == 0:
        return []
    results = vdb.search(query_vector, k=k)
    return results

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10)
)
def generate_response_with_groq(prompt: str) -> str:
    """Generates a response using Groq's Llama model."""
    if not groq_client:
        raise ValueError("Groq client not initialized. Check your GROQ_API_KEY.")
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config.GENERATIVE_MODEL_GROQ,
    )
    return chat_completion.choices[0].message.content

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10)
)
def generate_response_with_gemini(prompt: str) -> str:
    """Generates a response using Gemini's flash model."""
    response = gemini_client.models.generate_content(
        model=config.GENERATIVE_MODEL_GEMINI,
        contents=prompt
    )
    return response.text

def generate_rag_response(query: str, vdb: VectorDB, use_groq: bool = False) -> Dict[str, Any]:
    """Combines multimodal retrieval to generate a final AI response."""
    # 1. Vector Search (Multimodal)
    vector_results = retrieve_from_vector_db(query, vdb, k=6)
    
    text_context = []
    images_found = []
    tables_found = []
    
    for res in vector_results:
        if res["type"] == "text":
            text_context.append(f"[Source: Page {res['page']}] {res['content']}")
        elif res["type"] == "table":
            tables_found.append(res)
            text_context.append(f"[Source: Table Page {res['page']}] {res['content']}")
        elif res["type"] == "image":
            images_found.append(res)
            
    # 2. Build Prompt
    full_context = "\n\n".join(text_context)
    
def sanitize_mermaid_code(code: str) -> str:
    """Cleans Mermaid code to prevent syntax errors."""
    if not code:
        return ""
    
    # Remove any markdown code blocks if the LLM leaked them in
    code = re.sub(r'```mermaid|```', '', code).strip()
    
    # Ensure it starts with a valid graph type if missing
    if not any(code.startswith(t) for t in ["graph", "flowchart", "sequenceDiagram", "classDiagram"]):
        code = "graph TD\n" + code
        
    # Basic quoting fix for the most common error: unquoted labels with spaces or symbols
    # This is a simple regex that looks for IDs followed by labels in brackets
    # e.g., A[Some Label] -> A["Some Label"]
    code = re.sub(r'(\w+)\[(.*?)\]', r'\1["\2"]', code)
    code = re.sub(r'(\w+)\((.*?)\)', r'\1("\2")', code)
    code = re.sub(r'(\w+)\{(.*?)\}', r'\1{"\2"}', code)
    
    return code.strip()

def generate_rag_response(query: str, vdb: VectorDB, use_groq: bool = False) -> Dict[str, Any]:
    """Combines multimodal retrieval to generate a final AI response."""
    # 1. Vector Search (Multimodal)
    vector_results = retrieve_from_vector_db(query, vdb, k=6)
    
    text_context = []
    images_found = []
    tables_found = []
    
    for res in vector_results:
        if res["type"] == "text":
            text_context.append(f"[Source: Page {res['page']}] {res['content']}")
        elif res["type"] == "table":
            tables_found.append(res)
            text_context.append(f"[Source: Table Page {res['page']}] {res['content']}")
        elif res["type"] == "image":
            images_found.append(res)
            
    # 2. Build Prompt
    full_context = "\n\n".join(text_context)
    
    rag_prompt = f"""
    You are an expert Supply Chain Analyst. Answer the following question based ONLY on the provided document context.
    If the context doesn't contain the answer, say you don't know based on the document.
    
    SPECIAL INSTRUCTION:
    If the user asks for a "process," "flow," "cycle," or "relationship," OR if you think a diagram would help explain the answer, 
    provide a Mermaid.js flowchart code block.
    
    STRICT SYNTAX RULES:
    1. Always wrap labels in double quotes inside brackets: ID["Label Text"].
    2. Avoid using special characters like ( ) [ ] {{ }} directly in a Node ID.
    3. Format as:
    [MERMAID]
    graph TD
      A["Start"] --> B["Process (Step 1)"]
    [/MERMAID]
    
    CONTEXT:
    {full_context}
    
    QUESTION: {query}
    
    Provide a professional, formatted markdown response. Highlight critical supply chain entities.
    """
    
    # 3. Generate Final Answer
    try:
        if use_groq and groq_client:
            answer = generate_response_with_groq(rag_prompt)
        else:
            answer = generate_response_with_gemini(rag_prompt)
    except Exception as e:
        answer = f"Error generating response: {e}"
        
    # 4. Extract and Sanitize Mermaid Diagram Code
    mermaid_code = ""
    import re
    match = re.search(r'\[MERMAID\]\s*(.*?)\s*\[/MERMAID\]', answer, re.DOTALL | re.IGNORECASE)
    if match:
        raw_code = match.group(1).strip()
        mermaid_code = sanitize_mermaid_code(raw_code)
        # Clean up the answer text by removing the raw diagram code
        answer = re.sub(r'\[MERMAID\].*?\[/MERMAID\]', '', answer, flags=re.DOTALL | re.IGNORECASE).strip()
    
    return {
        "answer": answer,
        "mermaid_code": mermaid_code,
        "context": full_context,
        "images": images_found,
        "tables": tables_found,
        "vector_results": vector_results
    }

if __name__ == "__main__":
    pass
