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
import torch

# Page config
st.set_page_config(
    page_title="Document QA Demo",
    layout="wide"
)

st.title("📄 Document Q&A with FLAN-T5/GPT-2")
st.caption("Created by Uros Godnov")

@st.cache_resource
def load_flan_t5_model():
    """Load FLAN-T5-small model from Hugging Face"""
    try:
        # Try FLAN-T5 first, fallback to GPT-2 if sentencepiece fails
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            model_name = "google/flan-t5-small"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            return model, tokenizer, "flan-t5"
        except Exception as sentencepiece_error:
            st.warning(f"FLAN-T5 failed (sentencepiece issue): {sentencepiece_error}")
            st.info("Falling back to GPT-2 for text generation...")
            
            # Fallback to GPT-2 which doesn't need sentencepiece
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            model_name = "gpt2"
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # GPT-2 needs a pad token
            tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer, "gpt2"
            
    except Exception as e:
        st.error(f"Error loading any model: {e}")
        return None, None, None

def generate_flan_t5_answer(question, context_chunks, model, tokenizer, model_type, max_length=256):
    """Generate answer using FLAN-T5 or GPT-2 model"""
    
    if model is None or tokenizer is None:
        return "Model not available."
    
    try:
        # Combine chunks into context
        context = " ".join(context_chunks)
        
        # Anti-hallucination check
        question_keywords = set(question.lower().split())
        context_words = set(context.lower().split())
        overlap = question_keywords.intersection(context_words)
        
        if len(overlap) < 2 and len(question_keywords) > 3:
            return "Information not found in the document."
        
        # Different prompts for different models
        if model_type == "flan-t5":
            # FLAN-T5 prompt (instruction-following)
            prompt = f"{question}\n\nContext: {context}\n\nAnswer:"
        else:
            # GPT-2 prompt (completion-style)
            prompt = f"Question: {question}\n\nBased on this context: {context}\n\nAnswer: "
        
        # Truncate if needed
        if len(prompt) > 400:
            available_space = 400 - len(f"{question}\n\nContext: \n\nAnswer:")
            truncated_context = context[:available_space] + "..."
            if model_type == "flan-t5":
                prompt = f"{question}\n\nContext: {truncated_context}\n\nAnswer:"
            else:
                prompt = f"Question: {question}\n\nBased on this context: {truncated_context}\n\nAnswer: "
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        )
        
        # Different generation parameters for different models
        if model_type == "flan-t5":
            generation_params = {
                "max_length": max_length,
                "min_length": 15,
                "num_beams": 4,
                "early_stopping": True,
                "temperature": 0.7,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id
            }
        else:  # GPT-2
            generation_params = {
                "max_length": inputs['input_ids'].shape[1] + 100,  # Add tokens to input length
                "min_length": inputs['input_ids'].shape[1] + 20,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "no_repeat_ngram_size": 3,
            }
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_params)
        
        # Decode
        if model_type == "flan-t5":
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Clean up - just remove the prompt parts
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
        else:  # GPT-2
            # For GPT-2, we need to extract only the new tokens (the answer)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up GPT-2 response
            # Stop at first newline or period followed by newline (end of answer)
            for stop_seq in ['\n\n', '\nQuestion:', '\nQ:', '\n\n']:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
                    break
        
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
st.sidebar.subheader("Language Model")
st.sidebar.info("Tries FLAN-T5 first, falls back to GPT-2 if sentencepiece fails")

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
    help="Number of relevant chunks to use for answer generation"
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
            
            # Load language model
            with st.spinner('Loading language model...'):
                model, tokenizer, model_type = load_flan_t5_model()
            
            if model is None:
                st.error("Failed to load any language model.")
                st.stop()
            
            if model_type == "flan-t5":
                st.success(f'✅ Ready! ChromaDB collection with {len(chunks)} chunks and FLAN-T5 loaded!')
            else:
                st.success(f'✅ Ready! ChromaDB collection with {len(chunks)} chunks and GPT-2 loaded!')
                st.info("⚠️ Using GPT-2 fallback - instruction following may be limited.")
            
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
                placeholder="e.g., What is text embedding? Explain in 2 sentences.",
                help="Ask any question. Be specific with instructions: 'explain in 2 sentences', 'list 3 points', 'answer briefly', etc."
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
                    
                    with st.spinner('Generating answer...'):
                        answer = generate_flan_t5_answer(
                            query, 
                            retrieved_chunks, 
                            model, 
                            tokenizer, 
                            model_type,
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
    - 🤖 **FLAN-T5** with automatic **GPT-2 fallback** (no sentencepiece issues!)
    - 🎯 **Works without sentencepiece** - automatic model fallback
    - 📥 CSV export of Q&A results
    - 🚀 **No API keys required** - runs completely local!
    
    **Example Questions that work:**
    - "What is text embedding? Explain in 2 sentences."
    - "List 3 key benefits mentioned in the document."
    - "Summarize the main conclusion briefly."
    - "What are the risks? Answer in one paragraph."
    - "Who are the main stakeholders?"
    
    **Deployment Features:**
    - ✅ **No sentencepiece dependency issues** - automatic fallback to GPT-2
    - ✅ **Works on Streamlit Cloud** - handles compilation failures gracefully
    - ✅ **Fallback notification** - tells you which model is being used
    - ✅ **Robust deployment** - doesn't fail due to build issues
    
    **Optional for better performance:**
    ```
    sentencepiece
    ```
    (FLAN-T5 works better for instructions, but GPT-2 fallback ensures deployment success)
    
    **How it works:**
    1. Tries to load FLAN-T5 (best for instruction following)
    2. Falls back to GPT-2 if sentencepiece fails to compile
    3. ChromaDB finds relevant chunks using semantic search
    4. Model generates answers based on document content
    5. You get responses that follow your requirements (FLAN-T5) or good general answers (GPT-2)
    """)
    
    # Model loading test
    if st.button("🔄 Test Model Loading"):
        with st.spinner("Testing model loading..."):
            model, tokenizer, model_type = load_flan_t5_model()
            if model is not None:
                if model_type == "flan-t5":
                    st.success("✅ FLAN-T5 model loaded successfully!")
                else:
                    st.success("✅ GPT-2 fallback model loaded successfully!")
                    st.info("FLAN-T5 failed, but GPT-2 is working as backup.")
            else:
                st.error("❌ Failed to load any model")