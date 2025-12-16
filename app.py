"""
Streamlit frontend for GST Council RAG System with OSS120 LLM
Professional ChatGPT-style presentation
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "gst_council_documents"
PERSIST_DIRECTORY = "./chroma_db"
TOP_K_RESULTS = 8

OSS120_BASE_URL = "http://192.168.101.171:32003/v1"
OSS120_API_KEY = "dummy-key"
OSS120_MODEL = "oss120"
OSS120_MAX_TOKENS = 1500
OSS120_TEMPERATURE = 0.1


# ============================================================================
# CUSTOM CSS FOR CHATGPT-STYLE PRESENTATION
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for ChatGPT-like professional styling"""
    st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
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
    
    /* Answer container - ChatGPT style */
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
    
    .answer-content {
        padding: 1.2rem 1.5rem;
        color: #353740;
        font-size: 0.95rem;
        line-height: 1.75;
    }
    
    /* Source cards - Clean style */
    .source-item {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        margin: 0.6rem 0;
        overflow: hidden;
        transition: all 0.2s ease;
    }
    
    .source-item:hover {
        border-color: #10a37f;
        box-shadow: 0 2px 8px rgba(16, 163, 127, 0.1);
    }
    
    .source-header {
        background: #ffffff;
        padding: 0.7rem 1rem;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .source-title {
        font-weight: 600;
        color: #374151;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .source-badge {
        background: #10a37f;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .source-content {
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #4b5563;
        line-height: 1.6;
        max-height: 120px;
        overflow-y: auto;
    }
    
    .source-meta {
        padding: 0.5rem 1rem;
        background: #f3f4f6;
        font-size: 0.8rem;
        color: #6b7280;
        display: flex;
        gap: 1rem;
    }
    
    /* Info cards in sidebar */
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
    
    /* Success message */
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
    
    /* Custom divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 1.5rem 0;
    }
    
    /* Footer */
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
    
    /* Query section */
    .query-label {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_rag_system():
    """Initialize ChromaDB, Embedding Model, and OSS120 Client"""
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    llm_client = OpenAI(base_url=OSS120_BASE_URL, api_key=OSS120_API_KEY)
    return chroma_client, embedder, llm_client


@st.cache_resource
def load_collection(_client):
    """Load existing collection"""
    try:
        collection = _client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"❌ Collection not found! Please run rag.py first.")
        st.stop()


# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text"""
    words = text.lower().split()
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that'}
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords


def hybrid_search(collection, embedder, query: str, n_results: int = TOP_K_RESULTS) -> List[Dict]:
    """Hybrid search with keyword optimization"""
    query_embedding = embedder.encode([query])[0].tolist()
    query_keywords = extract_keywords(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results * 3, 30)
    )
    
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
        combined_score = (0.7 * keyword_score) + (0.2 * exact_score) + (0.1 * semantic_score)
        
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


# ============================================================================
# LLM GENERATION WITH OSS120
# ============================================================================

def generate_answer_with_oss120(llm_client, query: str, context_chunks: List[Dict]) -> str:
    """Generate a comprehensive, accurate answer using OSS120 model"""
    
    context_text = "\n\n---\n\n".join([
        f"[Source: {chunk['metadata']['source']}, Chunk {chunk['metadata']['chunk_id']}]\n{chunk['document']}"
        for chunk in context_chunks
    ])
    
    system_prompt = """You are an expert GST (Goods and Services Tax) analyst providing clear, professional answers.

RESPONSE FORMAT - Follow this structure:

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
            model=OSS120_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=OSS120_MAX_TOKENS,
            temperature=OSS120_TEMPERATURE,
        )
        
        answer = response.choices[0].message.content.strip()
        answer = clean_answer(answer)
        return answer
    
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"


def clean_answer(answer: str) -> str:
    """Clean and polish the answer"""
    
    # Remove common filler phrases
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
    
    # Capitalize first letter if needed
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    
    # Clean up extra whitespace
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = re.sub(r' {2,}', ' ', answer)
    
    return answer.strip()


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_answer(answer: str, query: str):
    """Display answer in ChatGPT-style format"""
    
    # Answer container header
    st.markdown("""
    <div class="answer-wrapper">
        <div class="answer-header">
            <div class="icon">🤖</div>
            <div class="title">GST Council Intelligence Response</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Streamlit's native markdown renderer for proper formatting
    # This will properly render headers, tables, bold, lists, etc.
    st.markdown(answer)


def display_sources(search_results: List[Dict], show_scores: bool):
    """Display source documents in clean cards"""
    
    st.markdown("---")
    st.markdown("#### 📚 Reference Sources")
    st.caption(f"Based on {len(search_results)} relevant document sections")
    
    for idx, result in enumerate(search_results):
        source_name = result['metadata']['source']
        chunk_id = result['metadata']['chunk_id']
        score = result['score']
        content = result['document']
        
        # Truncate content for preview
        preview = content[:300] + "..." if len(content) > 300 else content
        
        with st.expander(f"📄 {source_name} — Chunk {chunk_id}" + (f" (Score: {score:.2f})" if show_scores else ""), expanded=False):
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
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="GST Council RAG System",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ GST Council Meeting Intelligence</h1>
        <p>AI-Powered Document Analysis & Query Resolution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    try:
        chroma_client, embedder, llm_client = initialize_rag_system()
        collection = load_collection(chroma_client)
    except Exception as e:
        st.error("❌ Failed to initialize system. Please run rag.py first!")
        st.info(f"Error: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        doc_count = collection.count()
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{doc_count:,}</div>
            <div class="stat-label">Indexed Document Chunks</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        show_sources = st.checkbox("📚 Show source documents", value=True)
        show_scores = st.checkbox("📊 Show relevance scores", value=False)
        
        st.markdown("---")
        
        st.markdown("### 🔧 Configuration")
        st.caption(f"**Model:** {OSS120_MODEL.upper()}")
        st.caption(f"**Embeddings:** BGE-Large")
        st.caption(f"**Top Results:** {TOP_K_RESULTS}")
        
        st.markdown("---")
        
        if st.button("🔄 Refresh System", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Main content
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
    
    # Process query
    if search_button and query:
        if collection.count() == 0:
            st.warning("⚠️ No documents indexed. Please run rag.py first.")
        else:
            # Search
            with st.spinner("🔎 Searching documents..."):
                search_results = hybrid_search(collection, embedder, query, TOP_K_RESULTS)
            
            if not search_results:
                st.warning("No relevant information found in the database.")
            else:
                # Generate answer
                with st.spinner("🤖 Generating response..."):
                    answer = generate_answer_with_oss120(llm_client, query, search_results)
                
                # Success message
                st.markdown(f"""
                <div class="success-msg">
                    <span>✅</span>
                    <span><strong>Response generated</strong> from {len(search_results)} relevant sources</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Display answer using native Streamlit markdown
                display_answer(answer, query)
                
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
        Powered by ChromaDB • BGE-Large Embeddings • OSS120 LLM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
