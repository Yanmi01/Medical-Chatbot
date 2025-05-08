from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings



def load_pdf(file_path):
    loader = DirectoryLoader(file_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def chunk_data(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200) # Initialize the text splitter with a chunk size of 1000 characters and an overlap of 200 characters
    text_chunks = text_splitter.split_documents(extracted_data) # Split the documents into smaller chunks

    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

