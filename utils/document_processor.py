from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

class DocumentProcessor:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_documents(self, file_path):
        """Load documents based on file extension"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        return loader.load()

    def process_documents(self, documents):
        """Split documents into chunks and create embeddings"""
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents, store_name=None):
        """Create and save a vector store from documents"""
        chunks = self.process_documents(documents)
        vector_store = FAISS.from_documents(chunks, self.embedding_model)
        
        if store_name:
            vector_store.save_local(f"vector_stores/{store_name}")
            
        return vector_store
        
    def load_vector_store(self, store_name):
        """Load a previously saved vector store"""
        store_path = f"vector_stores/{store_name}"
        if not os.path.exists(store_path):
            return None
        return FAISS.load_local(store_path, self.embedding_model, allow_dangerous_deserialization=True)
