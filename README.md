# Medical Chatbot with Local LLM
A conversational AI assistant designed to provide medical information and answer health-related queries. Built with Python and natural language processing (NLP) techniques.

## Features
- **Local LLM Inference** using `CTransformers` (LLaMA/GGML-based)
- **LangChain**-powered Retrieval-Augmented Generation (RAG)
- **Flask Web Interface** for user interaction
- **Pinecone**-based vector search (optional)
- **Hugging Face Embeddings**

## How It Works

1. The user enters a medical query in the web app.
2. The query is converted into embeddings using a Hugging Face model.
3. The embeddings go to the vector database and retrieve relevant context.
4. The context and user query are passed to a **local LLaMA model**.
5. The LLM generates a response based on the prompt template.
6. The response is displayed in the chat interface.

## Setup Instructions

```bash
### 1. Clone the repository
git clone https://github.com/yourusername/Medical-Chatbot.git
cd Medical-Chatbot

### 2. Create and activate a virtual environment
conda create --name name python=3.10

### 3. Activate the Environment
conda activate name

### 4. Install Required Packages
conda install --file requirements.txt

### 5. Add your environment variables
Create a .env file in the root directory:
PINECONE_API_KEY="your_api_key"

### 6. Download and place your local model
Place your quantized .bin model (e.g., llama-2-7b-chat.ggmlv3.q4_0.bin) inside the model/ directory.

```

## Running the App Locally
```bash
python flask_app.py
```