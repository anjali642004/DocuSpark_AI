import streamlit as st
import time
import os
from src.processor import PDFProcessor
from src.embedding import EmbeddingManager
from src.chat import ChatManager
from src.config import Config
from langchain.schema import Document

st.set_page_config(page_title="PDF-RAG Chatbot", page_icon="🧠", layout="wide")

def initialize_session_state():
    """
    Initialize the Streamlit session state with the required components.
    Creates instances of the processor, embedding manager, and chat manager.
    """
    if "processor" not in st.session_state:
        st.session_state.processor = PDFProcessor()
        
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
        
    if "chat_manager" not in st.session_state:
        # Check if API key is available
        if not Config.is_valid():
            st.error("Missing API key. Please check your .env file.")
            return False
            
        st.session_state.chat_manager = ChatManager(Config.GOOGLE_API_KEY)
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "documents" not in st.session_state:
        st.session_state.documents = []
        
    return True

def process_documents(files_to_process):
    """
    Process the uploaded PDF documents.
    
    Args:
        files_to_process: List of uploaded PDF files or file paths
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        with st.spinner("Processing documents..."):
            all_documents = []
            
            # Check file sizes (2GB = 2 * 1024 * 1024 * 1024 bytes)
            max_size = 2 * 1024 * 1024 * 1024  # 2GB in bytes
            
            for file in files_to_process:
                # Handle both file objects and file paths
                if isinstance(file, str):  # File path
                    file_size = os.path.getsize(file)
                    file_name = os.path.basename(file)
                    
                    if file_size > max_size:
                        st.error(f"File '{file_name}' is too large ({file_size / (1024**3):.2f} GB). Maximum allowed size is 2GB.")
                        return False
                    
                    # Open file and process
                    with open(file, 'rb') as f:
                        documents = st.session_state.processor.process_document(f)
                        all_documents.extend(documents)
                else:  # File object (from uploader)
                    # Check file size
                    file.seek(0, 2)  # Seek to end
                    file_size = file.tell()  # Get file size
                    file.seek(0)  # Reset to beginning
                    
                    if file_size > max_size:
                        st.error(f"File '{file.name}' is too large ({file_size / (1024**3):.2f} GB). Maximum allowed size is 2GB.")
                        return False
                    
                    # Process each document using LangChain pipeline
                    documents = st.session_state.processor.process_document(file)
                    all_documents.extend(documents)
                
            # Store processed documents in session state
            st.session_state.documents = all_documents
            
            # Create embeddings for the documents
            success = st.session_state.embedding_manager.create_embeddings(all_documents)
            
            if success:
                # Connect the retriever to the chat manager
                st.session_state.chat_manager.set_retriever(
                    st.session_state.embedding_manager.retriever
                )
                st.success(f"Successfully processed {len(all_documents)} document chunks!")
                return True
            else:
                st.error("Failed to create embeddings.")
                return False
                
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

def main():
    """
    Main application function.
    Sets up the UI and handles user interactions.
    """
    # Initialize session state
    if not initialize_session_state():
        return
    
    st.header("📚🚀 DocuSpark AI – Upload Any Document, Get Instant Answers from Your PDFs 💡")
    
    # Sidebar for document upload and controls
    with st.sidebar:
        st.header("Upload Documents")
        
        # Custom file upload with 2GB support
        st.markdown("**📁 Upload PDF files (Max 2GB each)**")
        st.markdown("*Note: For files larger than 200MB, use the file path input below*")
        
        # Standard file uploader for smaller files
        uploaded_files = st.file_uploader(
            "Upload PDF files (Up to 200MB)", 
            type=['pdf'], 
            accept_multiple_files=True,
            help="Upload PDF files up to 200MB in size"
        )
        
        # File path input for larger files (2GB support)
        st.markdown("---")
        st.markdown("**For files larger than 200MB (up to 2GB):**")
        file_path = st.text_input(
            "Enter file path:",
            placeholder="C:/path/to/your/file.pdf",
            help="Enter the full path to your PDF file"
        )
        
        if file_path:
            if not file_path.lower().endswith('.pdf'):
                st.error("Please enter a valid PDF file path")
            elif not os.path.exists(file_path):
                st.error("File not found. Please check the path.")
            else:
                # Create a file-like object from the path
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > 2 * 1024 * 1024 * 1024:  # 2GB
                        st.error(f"File is too large ({file_size / (1024**3):.2f} GB). Maximum allowed size is 2GB.")
                    else:
                        st.success(f"File found: {os.path.basename(file_path)} ({file_size / (1024**2):.1f} MB)")
                        # Add to uploaded_files list
                        if 'large_files' not in st.session_state:
                            st.session_state.large_files = []
                        st.session_state.large_files.append(file_path)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Process button for all files
        all_files_to_process = []
        if uploaded_files:
            all_files_to_process.extend(uploaded_files)
        if 'large_files' in st.session_state and st.session_state.large_files:
            all_files_to_process.extend(st.session_state.large_files)
            
        if all_files_to_process:
            process_button = st.button("Process Documents")
            if process_button:
                process_documents(all_files_to_process)
                
        if st.session_state.documents:
            st.success(f"{len(st.session_state.documents)} chunks in memory")
            
            # Add a button to clear the conversation history
            if st.button("Clear Conversation"):
                st.session_state.messages = []
                st.session_state.chat_manager.reset_conversation()
                st.rerun()
            # Add a button to clear file chunks
            if st.button("Clear File Chunks"):
                st.session_state.documents = []
                if 'large_files' in st.session_state:
                    st.session_state.large_files = []
                if hasattr(st.session_state.embedding_manager, 'clear_embeddings'):
                    st.session_state.embedding_manager.clear_embeddings()
                st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if query := st.chat_input("Ask your question"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Check if documents have been uploaded and processed
        if not st.session_state.documents:
            with st.chat_message("assistant"):
                st.write("Please upload and process PDF documents first!")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload and process PDF documents first!"})
            return

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # If using direct retriever-based approach
                if hasattr(st.session_state.embedding_manager, 'retriever') and st.session_state.embedding_manager.retriever:
                    # Using LangChain's conversational retrieval chain
                    response = st.session_state.chat_manager.generate_response(query, [])
                else:
                    # Fallback to manual retrieval and response generation
                    relevant_docs = st.session_state.embedding_manager.search(query)
                    response = st.session_state.chat_manager.generate_response(query, relevant_docs)
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()