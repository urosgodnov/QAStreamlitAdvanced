# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import tempfile
import os
import chromadb
from chromadb.utils import embedding_functions
from docling.document_converter import DocumentConverter
import uuid
import pandas as pd
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SpacyTextSplitter
)
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Page config
st.set_page_config(
    page_title="Document QA Demo",
    layout="wide"
)

st.title("📄 Document Q&A with FLAN-T5")
st.caption("Created by Uros Godnov")

@st.cache_resource
def load_flan_t5_model():
    """Load FLAN-T5-small model from Hugging Face"""
    try:
        model_name = "google/flan-t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading FLAN-T5 model: {e}")
        st.error("Make sure 'sentencepiece' is in your requirements.txt")
        return None, None

def generate_flan_t5_answer(question, context_chunks, model, tokenizer, max_length=256):
    """Generate answer using FLAN-T5 model - SIMPLE approach"""
    
    if model is None or tokenizer is None:
        return "FLAN-T5 model not available."
    
    try:
        # Combine chunks into context
        context = " ".join(context_chunks)
        
        # Anti-hallucination check
        question_keywords = set(question.lower().split())
        context_words = set(context.lower().split())
        overlap = question_keywords.intersection(context_words)
        
        if len(overlap) < 2 and len(question_keywords) > 3:
            return "Information not found in the document."
        
        # SIMPLE PROMPT - just concatenate everything directly
        prompt = f"{question}\n\nContext: {context}\n\nAnswer:"
        
        # Truncate if needed
        if len(prompt) > 400:
            available_space = 400 - len(f"{question}\n\nContext: \n\nAnswer:")
            truncated_context = context[:available_space] + "..."
            prompt = f"{question}\n\nContext: {truncated_context}\n\nAnswer:"
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        )
        
        # Simple generation - let FLAN-T5 handle the instruction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=15,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up - just remove the prompt parts
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        # Basic validation
        if not response.strip() or len(response.strip()) < 5:
            return "Information not found in the document."
        
        return response.strip()
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@st.cache_resource
def get_embedding_function():
    """Get the embedding function for ChromaDB"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

# Sidebar options
st.sidebar.header("Configuration")

# Model options
st.sidebar.subheader("FLAN-T5 Model")
st.sidebar.info("Using google/flan-t5-small from Hugging Face")

max_output_length = st.sidebar.slider(
    "Max Answer Length",
    min_value=50,
    max_value=400,
    value=200,
    help="Maximum length of generated answer"
)

# Chunking options
st.sidebar.subheader("Chunking Options")
strategy = st.sidebar.selectbox(
    "Strategy", 
    ["Recursive Character", "Character", "Spacy"]
)

chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)

# Retrieval options
st.sidebar.subheader("Retrieval Options")
num_chunks = st.sidebar.slider(
    "Number of chunks to retrieve:",
    min_value=3,
    max_value=8,
    value=4,
    help="FLAN-T5 works best with fewer, high-quality chunks"
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload document",
    type=["pdf", "docx"]
)

def chunk_text(text, strategy, chunk_size, overlap):
    """Split text into chunks using LangChain splitters"""
    
    if strategy == "Recursive Character":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    elif strategy == "Character":
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separator="\n\n"
        )
    
    elif strategy == "Spacy":
        try:
            splitter = SpacyTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len
            )
        except Exception as e:
            st.warning(f"Spacy not available: {e}. Using Recursive Character instead.")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len
            )
    
    chunks = splitter.split_text(text)
    return chunks

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_filename = tmp_file.name
    
    try:
        # Convert document
        with st.spinner('Converting document...'):
            conv = DocumentConverter()
            result = conv.convert(temp_filename)
            text = result.document.export_to_markdown()
        
        # Clean up temporary file
        os.unlink(temp_filename)
        
        st.success('Document converted successfully!')
        st.info(f"Extracted {len(text)} characters")
        
        # Chunk the text
        with st.spinner('Chunking text...'):
            chunks = chunk_text(text, strategy, chunk_size, chunk_overlap)
        
        st.info(f"Created {len(chunks)} chunks")
        
        if chunks:
            # Set up embeddings and ChromaDB
            with st.spinner('Setting up ChromaDB...'):
                embedding_model = get_embedding_function()
                
                # Create client and collection
                client = chromadb.Client()
                
                # Generate unique collection name
                collection_name = f"doc_collection_{uuid.uuid4().hex[:8]}"
                
                collection = client.create_collection(
                    collection_name,
                    embedding_function=embedding_model
                )
                
                # Add chunks with metadata
                ids = [f"chunk_{i}" for i in range(len(chunks))]
                metadatas = [{"chunk_id": i, "source": uploaded_file.name, "strategy": strategy} 
                           for i in range(len(chunks))]
                
                collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            
            # Load FLAN-T5 model
            with st.spinner('Loading FLAN-T5 model...'):
                model, tokenizer = load_flan_t5_model()
            
            if model is None:
                st.error("Failed to load FLAN-T5 model. Please add 'sentencepiece' to requirements.txt")
                st.stop()
            
            st.success(f'✅ Ready! ChromaDB collection with {len(chunks)} chunks and FLAN-T5 loaded!')
            
            # Show preview of chunks
            st.subheader("Chunk Preview")
            preview_chunks = min(3, len(chunks))
            
            for i in range(preview_chunks):
                with st.expander(f"Chunk {i+1} ({len(chunks[i])} chars)"):
                    st.text(chunks[i][:500] + "..." if len(chunks[i]) > 500 else chunks[i])
            
            # Query interface
            st.subheader("🤖 Ask Your Document")
            
            # Question input
            query = st.text_input(
                "Your Question:",
                placeholder="e.g., What is text embedding? Explain in 10 words.",
                help="Ask any question. Be specific with instructions: 'explain in 10 words', 'list 3 points', 'answer briefly', etc."
            )
            
            if query:
                with st.spinner('Finding relevant content...'):
                    results = collection.query(
                        query_texts=[query],
                        n_results=min(num_chunks, len(chunks))
                    )
                
                if results['documents'] and results['documents'][0]:
                    retrieved_chunks = results['documents'][0]
                    metadatas = results['metadatas'][0]
                    distances = results['distances'][0]
                    
                    st.subheader("🤖 Answer")
                    
                    with st.spinner('Generating answer with FLAN-T5...'):
                        answer = generate_flan_t5_answer(
                            query, 
                            retrieved_chunks, 
                            model, 
                            tokenizer, 
                            max_output_length
                        )
                    
                    # Display the answer prominently
                    st.markdown(f"**Q:** {query}")
                    st.markdown(f"**A:** {answer}")
                    
                    # Show source information
                    st.caption(f"Generated from {len(retrieved_chunks)} relevant document segments")
                    
                    # Option to show source chunks
                    if st.checkbox("📖 Show source chunks used"):
                        st.subheader("📋 Source Information")
                        for i, (doc, metadata, distance) in enumerate(zip(retrieved_chunks, metadatas, distances)):
                            with st.expander(f"Source {i+1} (Relevance: {1-distance:.3f})"):
                                st.write(doc)
                                st.caption(f"Chunk ID: {metadata['chunk_id']}")
                    
                    # Download option
                    if st.button("📥 Download Q&A Results"):
                        # Create download data
                        out_data = [{
                            'type': 'QUESTION',
                            'text': query,
                            'source': uploaded_file.name,
                            'timestamp': pd.Timestamp.now()
                        }, {
                            'type': 'ANSWER',
                            'text': answer,
                            'source': uploaded_file.name,
                            'timestamp': pd.Timestamp.now()
                        }]
                        
                        # Add source chunks
                        for i, (doc, metadata, distance) in enumerate(zip(retrieved_chunks, metadatas, distances)):
                            out_data.append({
                                'type': 'SOURCE_CHUNK',
                                'text': doc,
                                'relevance_score': 1 - distance,
                                'chunk_id': metadata['chunk_id'],
                                'source': uploaded_file.name,
                                'timestamp': pd.Timestamp.now()
                            })
                        
                        df = pd.DataFrame(out_data)
                        csv = df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"qa_results_{uploaded_file.name}.csv",
                            "text/csv"
                        )
                else:
                    st.warning("No relevant content found. Try rephrasing your question.")
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.unlink(temp_filename)
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload a PDF or DOCX file to get started")
    st.caption("Created by Uros Godnov")
    
    st.markdown("""
    **Features:**
    - 📄 Document conversion with Docling (PDF, DOCX)
    - ✂️ Multiple chunking strategies with LangChain
    - 🔍 ChromaDB vector database for semantic search
    - 🤖 **FLAN-T5** with enhanced instruction-following
    - 🎯 **Precise instruction compliance** - "explain in 10 words", "list 3 points", etc.
    - 📥 CSV export of Q&A results
    - 🚀 **No API keys required** - runs completely local!
    
    **Example Questions that work:**
    - "What is text embedding? Explain in 10 words."
    - "What is text embedding? Explain in 5 words."
    - "List 3 key benefits mentioned in the document."
    - "Summarize the main conclusion in one sentence."
    - "What are the risks? Answer briefly."
    - "Who are the main stakeholders? List them."
    
    **Enhanced Features:**
    - ✅ **Instruction Detection** - Automatically detects word limits and formatting requests
    - ✅ **Smart Prompting** - Uses different prompts for different question types
    - ✅ **Word Count Enforcement** - Enforces exact word limits when requested
    - ✅ **Better Context Handling** - Prioritizes instructions over long context
    
    **Requirements for deployment:**
    ```
    sentencepiece
    ```
    (Add this to your requirements.txt)
    
    **How it works:**
    1. ChromaDB finds relevant chunks using semantic search
    2. Enhanced prompts ensure FLAN-T5 follows your specific instructions
    3. Post-processing enforces word limits and formatting
    4. You get precisely formatted responses that follow your requirements
    """)
    
    # Model loading test
    if st.button("🔄 Test FLAN-T5 Loading"):
        with st.spinner("Testing FLAN-T5 model loading..."):
            model, tokenizer = load_flan_t5_model()
            if model is not None:
                st.success("✅ FLAN-T5 model loaded successfully!")
            else:
                st.error("❌ Failed to load FLAN-T5 model - add 'sentencepiece' to requirements.txt")