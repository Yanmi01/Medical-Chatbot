from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from src.prompt import prompt_template
import pinecone
import time
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from dotenv import load_dotenv
import os


app = Flask(__name__)

load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key = pinecone_api_key)

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot-with-llama2" 

index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    index=index, 
    embedding=embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

# use this code only if you have downloaded the model and stored it in the model folder.
# you can also go download the model and save it before using this code below
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          "context_length": 2048,  
                          'temperature':0.8,
                          "local_files_only": True})

# use this commented code rather than the one above if you haven't downloaded the model before. this will help download the model locally

# llm = CTransformers(
#     model="TheBloke/Llama-2-7B-Chat-GGML",  # Folder path, not the full .bin file
#     model_file="llama-2-7b-chat.ggmlv3.q4_0.bin",  # Actual model filename
#     model_type="llama",
#     config={
#         "max_new_tokens": 512,
#         "temperature": 0.8,
#     }
# )


retriever = vector_store.as_retriever()

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa.invoke({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
