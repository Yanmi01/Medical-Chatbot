from src.helper import load_pdf, chunk_data, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dot_env import load_dotenv
import os
import time


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

extracted_data = load_pdf("data/")
text_chunks = chunk_data(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key = pinecone_api_key)

index_name = "medical-chatbot-with-llama2" 

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="enabled",  # Defaults to "disabled"
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    index=index, 
    embedding=embeddings)

vector_store.add_documents(text_chunks)



