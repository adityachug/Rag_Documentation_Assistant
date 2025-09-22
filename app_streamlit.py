import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from vector_store.faiss_store import FaissStore

# Page config
st.set_page_config(
    page_title="RAG Documentation Assistant",
    page_icon="🤖",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system"""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("❌ GOOGLE_API_KEY not found!")
            return None, None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        store = FaissStore()
        return model, store
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None, None

# System prompt
SYSTEM_PROMPT = """You are an assistant that answers only using the provided documentation excerpts. 
Always cite the exact attribute name(s) used in your answer in square brackets like [attribute: workspace_id]. 
If the documentation does not contain an answer, respond: "I don't know — please check the source."

DOCUMENTS:
{documents}

USER QUESTION: {query}

INSTRUCTIONS:
- Use only the DOCUMENTS to answer.
- Provide a confidence label (High/Medium/Low).
- After the answer, list the sources used with their attribute names."""

def build_prompt(query, retrieved):
    docs_text = ""
    for i, r in enumerate(retrieved, start=1):
        m = r["meta"]
        docs_text += f"{i}) attribute: {m['attribute']}\n   type: {m['type']}\n   description: {m['description']}\n\n"
    return SYSTEM_PROMPT.format(documents=docs_text, query=query)

def answer_query(query, model, store, k=5):
    try:
        retrieved = store.search(query, k=k)
        
        if not retrieved:
            return {"text": "I don't know — please check the source.", "sources": [], "confidence": "Low"}
        
        prompt = build_prompt(query, retrieved)
        response = model.generate_content(prompt)
        
        # Calculate confidence
        avg_score = sum([r['score'] for r in retrieved]) / len(retrieved)
        if avg_score < 0.3:
            confidence = "High"
        elif avg_score < 1.0:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        sources = [{"attribute": r["meta"]["attribute"], "type": r["meta"].get("type", ""), "score": r["score"]} for r in retrieved]
        
        return {"text": response.text, "sources": sources, "confidence": confidence}
        
    except Exception as e:
        return {"text": f"Error: {str(e)}", "sources": [], "confidence": "Low"}

# Main app
def main():
    st.title("🤖 RAG Documentation Assistant")
    st.markdown("Ask questions about your API documentation!")
    
    # Initialize system
    model, store = initialize_rag_system()
    if model is None or store is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        k_docs = st.slider("Documents to retrieve", 1, 10, 5)
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            # REMOVED st.rerun() - not needed, Streamlit updates automatically
        
        st.header("💡 Examples")
        examples = ["What is workspace_id?", "How to create a ticket?", "Show me ticket attributes"]
        for example in examples:
            if st.button(example, key=f"example_{example}"):  # Added unique keys
                st.session_state.current_query = example
    
    # Chat interface with unique key
    if hasattr(st.session_state, 'current_query'):
        query = st.text_input("Your question:", value=st.session_state.current_query, key="user_input")
        del st.session_state.current_query
    else:
        query = st.text_input("Your question:", key="user_input")
    
    # FIXED: Only trigger on button click, with processing flag to prevent multiple calls
    if st.button("🚀 Ask") and not st.session_state.processing:
        if query.strip():
            st.session_state.processing = True  # Prevent multiple calls
            
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("Generating answer..."):
                result = answer_query(query, model, store, k_docs)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": result["text"],
                "sources": result["sources"],
                "confidence": result["confidence"]
            })
            
            # Clear the input field
            if "user_input" in st.session_state:
                del st.session_state.user_input
            
            st.session_state.processing = False  # Reset processing flag
            # REMOVED st.rerun() - Streamlit handles updates automatically
    
    # Display chat
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                confidence = message.get("confidence", "Unknown")
                if confidence == "High":
                    st.success(f"🎯 Confidence: {confidence}")
                elif confidence == "Medium":
                    st.warning(f"🎯 Confidence: {confidence}")
                else:
                    st.error(f"🎯 Confidence: {confidence}")
                
                sources = message.get("sources", [])
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, source in enumerate(sources[:3], 1):
                            st.write(f"**{i}. {source['attribute']}** ({source['type']}) - Score: {source['score']:.3f}")

if __name__ == "__main__":
    main()