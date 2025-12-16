"""
GST Council RAG System - Merged Single File Application
Compatible with Streamlit Community Cloud
"""

import streamlit as st
from typing import List, Dict
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "gst_council_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 5

LLM_BASE_URL = "http://192.168.101.171:32003/v1"
LLM_API_KEY = "dummy-key"
LLM_MODEL = "oss120"
LLM_MAX_TOKENS = 1500
LLM_TEMPERATURE = 0.1


# ============================================================================
# SECTION 1: BACKEND HELPER FUNCTIONS
# ============================================================================

def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\{\}\%\₹]', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 30:
            chunks.append(chunk)
    return chunks


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text"""
    words = text.lower().split()
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that'}
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file content"""
    try:
        import PyPDF2
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception:
        return ""


def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file content"""
    try:
        return file_content.decode('utf-8')
    except:
        try:
            return file_content.decode('latin-1')
        except:
            return ""


def process_uploaded_files(files, embedder, collection) -> int:
    """Process uploaded files and add to collection"""
    total_chunks = 0
    
    for file in files:
        try:
            file_content = file.read()
            file_name = file.name
            
            if file_name.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_content)
            elif file_name.lower().endswith('.txt'):
                text = extract_text_from_txt(file_content)
            else:
                continue
            
            if not text or len(text) < 100:
                continue
            
            cleaned_text = preprocess_text(text)
            chunks = chunk_text(cleaned_text)
            
            if not chunks:
                continue
            
            embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
            
            ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {"source": file_name, "chunk_id": i}
                for i in range(len(chunks))
            ]
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            total_chunks += len(chunks)
        except Exception as e:
            st.warning(f"Error processing {file.name}: {e}")
            continue
    
    return total_chunks


def hybrid_search(collection, embedder, query: str, n_results: int = TOP_K_RESULTS) -> List[Dict]:
    """Hybrid search with keyword optimization"""
    try:
        query_embedding = embedder.encode([query])[0].tolist()
        query_keywords = extract_keywords(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results * 3, 30)
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        scored_results = []
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            doc_keywords = set(extract_keywords(doc))
            query_keyword_set = set(query_keywords)
            
            keyword_matches = len(doc_keywords.intersection(query_keyword_set))
            keyword_score = keyword_matches / max(len(query_keyword_set), 1)
            exact_matches = sum(1 for kw in query_keywords if kw in doc.lower())
            exact_score = exact_matches / max(len(query_keywords), 1)
            semantic_score = 1 - distance
            combined_score = (0.5 * keyword_score) + (0.2 * exact_score) + (0.3 * semantic_score)
            
            scored_results.append({
                'document': doc,
                'metadata': metadata,
                'score': combined_score,
                'keyword_score': keyword_score,
                'exact_score': exact_score,
                'semantic_score': semantic_score
            })
        
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:n_results]
    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def generate_answer(llm_client, query: str, context_chunks: List[Dict]) -> str:
    """Generate answer using LLM"""
    if not context_chunks:
        return "No relevant information found in the documents."
    
    context_text = "\n\n---\n\n".join([
        f"[Source: {chunk['metadata']['source']}, Chunk {chunk['metadata']['chunk_id']}]\n{chunk['document']}"
        for chunk in context_chunks
    ])
    
    system_prompt = """You are an expert GST (Goods and Services Tax) analyst providing clear, professional answers.

RESPONSE FORMAT:

### Summary
Write 2-3 sentences summarizing the key finding.

### Key Details
Present the main information using:
- **Bold** for important terms, rates, percentages
- Bullet points for listing items
- Clear, complete sentences

### Source References
Mention which GST Council meeting(s) or document(s) contain this information.

### Conclusion
One sentence summarizing the practical implication.

IMPORTANT RULES:
1. Answer ONLY from the provided context
2. Use proper markdown formatting
3. Be specific with numbers (₹X,XXX, X%)
4. Complete all sentences
5. Never say "based on the context" - just provide the answer directly
6. Keep paragraphs short and readable"""

    user_prompt = f"""Question: {query}

Reference Documents:
{context_text}

Provide a clear, well-structured answer following the format specified."""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        answer = response.choices[0].message.content.strip()
        return clean_answer(answer)
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"


def clean_answer(answer: str) -> str:
    """Clean the answer"""
    filler_patterns = [
        r"^Based on the provided context,?\s*",
        r"^According to the documents?,?\s*",
        r"^The context mentions that\s*",
        r"^From the information given,?\s*",
        r"^Based on the context above,?\s*",
        r"^According to the provided information,?\s*",
    ]
    for pattern in filler_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = re.sub(r' {2,}', ' ', answer)
    return answer.strip()


# ============================================================================
# SECTION 2: CACHED LOADER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_embedder():
    """Load embedding model"""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Failed to load embedder: {e}")
        return None


@st.cache_resource
def load_llm_client():
    """Load LLM client"""
    try:
        from openai import OpenAI
        return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    except Exception as e:
        st.warning(f"LLM client not available: {e}")
        return None


def get_chroma_collection():
    """Get or create ChromaDB collection (in-memory for Streamlit Cloud)"""
    if 'chroma_client' not in st.session_state:
        try:
            import chromadb
            st.session_state.chroma_client = chromadb.Client()
        except Exception as e:
            st.error(f"Failed to create ChromaDB client: {e}")
            return None
    
    if 'collection' not in st.session_state:
        try:
            st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.error(f"Failed to create collection: {e}")
            return None
    
    return st.session_state.collection


# ============================================================================
# SECTION 3: CUSTOM CSS
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for professional styling"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #10a37f 0%, #1a7f64 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(16, 163, 127, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
        font-weight: 400;
    }
    
    .answer-wrapper {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        margin: 1rem 0;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .answer-header {
        background: #f7f7f8;
        padding: 0.8rem 1.2rem;
        border-bottom: 1px solid #e5e5e5;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    .answer-header .icon {
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #10a37f 0%, #1a7f64 100%);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.9rem;
    }
    
    .answer-header .title {
        font-weight: 600;
        color: #202123;
        font-size: 0.95rem;
    }
    
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.5rem;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #10a37f;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.2rem;
    }
    
    .success-msg {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #065f46;
        font-size: 0.9rem;
    }
    
    .footer {
        text-align: center;
        padding: 1.2rem;
        background: #f9fafb;
        border-radius: 10px;
        margin-top: 2rem;
        color: #6b7280;
        font-size: 0.85rem;
        border: 1px solid #e5e7eb;
    }
    
    .query-label {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SECTION 4: DISPLAY FUNCTIONS
# ============================================================================

def display_answer(answer: str):
    """Display answer in ChatGPT-style format"""
    st.markdown("""
    <div class="answer-wrapper">
        <div class="answer-header">
            <div class="icon">🤖</div>
            <div class="title">GST Council Intelligence Response</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(answer)


def display_sources(search_results: List[Dict], show_scores: bool):
    """Display source documents"""
    st.markdown("---")
    st.markdown("#### 📚 Reference Sources")
    st.caption(f"Based on {len(search_results)} relevant document sections")
    
    for idx, result in enumerate(search_results):
        source_name = result['metadata']['source']
        chunk_id = result['metadata']['chunk_id']
        score = result['score']
        content = result['document']
        preview = content[:300] + "..." if len(content) > 300 else content
        
        label = f"📄 {source_name} — Chunk {chunk_id}"
        if show_scores:
            label += f" (Score: {score:.2f})"
        
        with st.expander(label, expanded=False):
            st.markdown(f"**Document:** {source_name}")
            st.markdown(f"**Section:** Chunk {chunk_id}")
            if show_scores:
                cols = st.columns(4)
                cols[0].metric("Combined", f"{result['score']:.2f}")
                cols[1].metric("Keyword", f"{result['keyword_score']:.2f}")
                cols[2].metric("Exact", f"{result['exact_score']:.2f}")
                cols[3].metric("Semantic", f"{result['semantic_score']:.2f}")
            st.markdown("---")
            st.markdown("**Content:**")
            st.text(preview)


# ============================================================================
# SECTION 5: MAIN STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="GST Council RAG System",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ GST Council Meeting Intelligence</h1>
        <p>AI-Powered Document Analysis & Query Resolution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load components with graceful fallback
    embedder = load_embedder()
    llm_client = load_llm_client()
    collection = get_chroma_collection()
    
    # Check initialization
    if embedder is None:
        st.error("❌ Failed to initialize embedding model. Please refresh the page.")
        st.stop()
    
    if collection is None:
        st.error("❌ Failed to initialize vector database. Please refresh the page.")
        st.stop()
    
    # Get document count safely
    try:
        doc_count = collection.count()
    except:
        doc_count = 0
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{doc_count:,}</div>
            <div class="stat-label">Indexed Document Chunks</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Document Upload Section
        st.markdown("### 📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("📥 Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    chunks_added = process_uploaded_files(uploaded_files, embedder, collection)
                if chunks_added > 0:
                    st.success(f"✅ Added {chunks_added} chunks")
                    st.rerun()
                else:
                    st.warning("No valid content found in files")
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        show_sources = st.checkbox("📚 Show source documents", value=True)
        show_scores = st.checkbox("📊 Show relevance scores", value=False)
        
        st.markdown("---")
        
        st.markdown("### 🔧 Configuration")
        st.caption(f"**Model:** {LLM_MODEL.upper()}")
        st.caption(f"**Embeddings:** MiniLM-L6-v2")
        st.caption(f"**Top Results:** {TOP_K_RESULTS}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_resource.clear()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("🗑️ Clear DB", use_container_width=True):
                if 'collection' in st.session_state:
                    try:
                        st.session_state.chroma_client.delete_collection(COLLECTION_NAME)
                        del st.session_state['collection']
                        st.rerun()
                    except:
                        pass
    
    # Main content area
    if doc_count == 0:
        st.info("👆 **Get Started:** Upload PDF or TXT documents using the sidebar to begin.")
        st.markdown("""
        ### How to use:
        1. **Upload** your GST Council meeting documents (PDF or TXT) using the sidebar
        2. **Click** "Process Documents" to index them
        3. **Ask** questions about the content
        
        ### Supported formats:
        - PDF files (.pdf)
        - Text files (.txt)
        """)
    else:
        st.markdown('<p class="query-label">💬 Ask a question about GST Council decisions</p>', unsafe_allow_html=True)
        
        query = st.text_area(
            "Enter your question",
            placeholder="Example: What are the GST rates for parking charges and common facility charges in residential complexes?",
            height=80,
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button and query:
            # Search
            with st.spinner("🔎 Searching documents..."):
                search_results = hybrid_search(collection, embedder, query, TOP_K_RESULTS)
            
            if not search_results:
                st.warning("No relevant information found in the database.")
            else:
                # Generate answer
                with st.spinner("🤖 Generating response..."):
                    if llm_client:
                        answer = generate_answer(llm_client, query, search_results)
                    else:
                        # Fallback when LLM is not available
                        answer = "### Relevant Excerpts Found\n\n"
                        for i, r in enumerate(search_results[:3], 1):
                            answer += f"**{i}. From {r['metadata']['source']}:**\n\n"
                            answer += f"{r['document'][:300]}...\n\n---\n\n"
                
                # Success message
                st.markdown(f"""
                <div class="success-msg">
                    <span>✅</span>
                    <span><strong>Response generated</strong> from {len(search_results)} relevant sources</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Display answer
                display_answer(answer)
                
                # Display sources
                if show_sources:
                    display_sources(search_results, show_scores)
    
    # Example queries
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        **Rate Changes:**
        - What are the GST rate changes for automobiles?
        - What is the GST rate on textile products?
        
        **Exemptions:**
        - What exemptions are provided for small businesses?
        - Are there any GST exemptions for healthcare services?
        
        **Real Estate:**
        - What about parking charges in residential complexes?
        - How is GST applied to construction services?
        
        **Compliance:**
        - What are the recent compliance requirements?
        - What are the input tax credit rules?
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <strong>GST Council Meeting Intelligence System</strong><br>
        Powered by ChromaDB • Sentence Transformers • LLM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

