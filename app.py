import streamlit as st
import os
import pandas as pd
import streamlit.components.v1 as html
import config
from pdf_parser import extract_text_from_pdf, extract_tables_from_pdf, extract_images_from_pdf, chunk_text
from embeddings import VectorDB, get_text_embeddings, get_table_embeddings, get_image_embedding
from retrieval import generate_rag_response
import numpy as np
import json
import time

# Page Configuration
st.set_page_config(page_title="Multimodal Supply Chain Intelligence", page_icon="🔗", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #1e2130;
        border-radius: 4px 4px 0px 0px; padding: 10px 20px;
    }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Sidebar - Secure Configuration
with st.sidebar:
    st.title("🛡️ System Control")
    st.write("---")
    
    # API Status Check
    gemini_ready = config.GEMINI_API_KEY and config.GEMINI_API_KEY != "YOUR_GEMINI_API_KEY"
    groq_ready = config.GROQ_API_KEY and config.GROQ_API_KEY != "YOUR_GROQ_API_KEY"
    
    st.subheader("🔑 API Connectivity")
    if gemini_ready:
        st.success("✅ Gemini (Embeddings): Active")
    else:
        st.error("❌ Gemini (Embeddings): Missing")
        
    if groq_ready:
        st.success("✅ Groq (Reasoning): Active")
    else:
        st.info("ℹ️ Groq (Reasoning): Not Configured")

    # Collapsible Advanced Settings for Key Management
    with st.expander("🔒 Change API Credentials", expanded=False):
        st.caption("Editing these will override your .env settings for this session.")
        new_google_key = st.text_input("New Gemini API Key", type="password")
        new_groq_key = st.text_input("New Groq API Key", type="password")
        
        if st.button("🔄 Update & Reload"):
            if new_google_key: config.GEMINI_API_KEY = new_google_key
            if new_groq_key: config.GROQ_API_KEY = new_groq_key
            st.rerun()

    st.write("---")
    st.subheader("🤖 Reasoning Engine")
    use_groq = st.toggle("Use Groq (Llama 3.3)", value=config.USE_GROQ if groq_ready else False, 
                         disabled=not groq_ready,
                         help="Fastest responses with Groq Llama 3.3")
    
    st.write("---")
    uploaded_file = st.file_uploader("Upload Supply Chain PDF", type="pdf", help="Max size: 200MB")
    
    if uploaded_file and st.button("🚀 Process Document"):
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            with st.status("🧠 Analyzing Document...", expanded=True) as status:
                st.write("Extracting Text, Tables, and Images...")
                text_data = extract_text_from_pdf(pdf_path)
                table_data = extract_tables_from_pdf(pdf_path)
                image_data = extract_images_from_pdf(pdf_path, config.TEMP_IMAGE_DIR)
                chunks = chunk_text(text_data)
                
                if not chunks:
                    st.error("No content found in the PDF. Please upload a valid document.")
                    st.stop()

                st.session_state.text_data = text_data
                st.session_state.table_data = table_data
                st.session_state.image_data = image_data
                st.session_state.chunks = chunks
                
                st.write("Encoding Multimodal Data (Gemini Embedding 2)...")
                vdb = VectorDB()
                # Text Embeddings
                text_contents = [c["content"] for c in chunks]
                text_embs = get_text_embeddings(text_contents)
                if text_embs.size == 0:
                    st.error("Failed to generate text embeddings. Check your Gemini API Key.")
                    st.stop()
                vdb.add_vectors(text_embs, chunks)
                
                # Table Embeddings
                if table_data:
                    table_embs = get_table_embeddings(table_data)
                    if table_embs.size > 0:
                        vdb.add_vectors(table_embs, [{"type": "table", "content": json.dumps(t["table"]), "page": t["page"]} for t in table_data])
                
                # Image Embeddings
                if image_data:
                    for img in image_data:
                        img_emb = get_image_embedding(img["path"])
                        vdb.add_vectors(np.array([img_emb]), [{"type": "image", "content": img["filename"], "path": img["path"], "page": img["page"]}])
                
                vdb.save(config.FAISS_INDEX_PATH)
                st.session_state.vdb = vdb
                status.update(label="✅ Multimodal Analysis Complete!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error during processing: {e}")
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

# Main Interface
st.title("🔗 Multimodal Supply Chain Intelligence")
st.markdown("Unlock deep insights from **Text, Tables, and Images** using **Gemini Embedding 2**.")

# Dashboard Stats
if "text_data" in st.session_state:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 Pages", len(st.session_state.text_data))
    m2.metric("🔡 Text Chunks", len(st.session_state.chunks))
    m3.metric("📊 Tables", len(st.session_state.table_data))
    m4.metric("🖼️ Images", len(st.session_state.image_data))

# Tabs for Insights and Data
tab1, tab2, tab3 = st.tabs(["💡 Multimodal Insights", "📊 Table Gallery", "🖼️ Image Gallery"])

def render_mermaid(mermaid_code: str):
    """Renders a Mermaid diagram in Streamlit with height management."""
    if not mermaid_code or len(mermaid_code) < 10:
        return
    
    # Simple height calculation based on lines of code
    num_lines = len(mermaid_code.split('\n'))
    height = max(200, min(800, num_lines * 45))
    
    html_code = f"""
    <div class="mermaid" style="display: flex; justify-content: center; background-color: transparent;">
        {mermaid_code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: true, 
            theme: 'dark',
            securityLevel: 'loose',
            flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}
        }});
    </script>
    <style>
        .mermaid {{ font-family: 'Inter', sans-serif !important; overflow: visible; }}
    </style>
    """
    html.html(html_code, height=height)

with tab1:
    query = st.text_input("Ask about the Supply Chain:", placeholder="e.g., 'Compare the risk factors across all manufacturing plants.'")
    
    if query:
        if "vdb" in st.session_state:
            with st.spinner(f"Reasoning with {'Groq' if use_groq else 'Gemini'}..."):
                result = generate_rag_response(query, st.session_state.vdb, use_groq=use_groq)
                
                st.markdown("### AI Analyst Response")
                st.markdown(result["answer"])
                
                if result.get("mermaid_code"):
                    st.markdown("---")
                    st.markdown("#### 🔄 Visual Summary / Process Flow")
                    render_mermaid(result["mermaid_code"])
                
                if result["images"]:
                    st.markdown("---")
                    st.markdown("#### 🖼️ Supporting Visual Evidence")
                    cols = st.columns(min(len(result["images"]), 4))
                    for idx, img in enumerate(result["images"][:4]):
                        with cols[idx]:
                            st.image(img["path"], caption=f"Evidence from Page {img['page']}")
                
                with st.expander("📌 Source Content (Text & Tables)"):
                    st.text(result["context"])
        else:
            st.info("Please upload and process a Supply Chain document in the sidebar first.")

with tab2:
    if "table_data" in st.session_state and st.session_state.table_data:
        for idx, table in enumerate(st.session_state.table_data):
            st.markdown(f"**Table {idx+1} (Page {table['page']})**")
            st.dataframe(pd.DataFrame(table["table"]), use_container_width=True)
    else:
        st.info("No tables detected in the document.")

with tab3:
    if "image_data" in st.session_state and st.session_state.image_data:
        cols = st.columns(3)
        for idx, img in enumerate(st.session_state.image_data):
            with cols[idx % 3]:
                st.image(img["path"], caption=f"Page {img['page']}: {img['filename']}")
    else:
        st.info("No images/diagrams detected in the document.")
