Local RAG Chatbot with Personas

This project is an end-to-end, containerized RAG (Retrieval-Augmented Generation) application. It allows you to "chat" with your PDF documents using local, open-source LLMs.

The application is built with a FastAPI backend that exposes endpoints for uploading documents and asking questions. It features a dynamic "Persona" system, allowing the LLM to adopt different personalities (e.g., a helpful assistant, a meticulous research scientist) based on the user's needs.

The entire stack is containerized using Docker Compose, making it reproducible, scalable, and easy to run with a single command.

Features

Chat With Your PDFs: Upload any PDF and ask questions about its content.

Dynamic Personas: Switch the LLM's personality and response style on the fly.

Default: A general-purpose helpful assistant.

Scientific: A formal, meticulous research assistant that answers only from the text.

Company Analyst: A savvy business analyst to help you understand company documents or job listings.

100% Local & Open-Source: Runs entirely on your machine using Ollama. No API keys needed.

Containerized Stack: The FastAPI app and the Ollama LLM server run in isolated, networked containers.

Persistent Vector Storage: PDF embeddings are saved to a persistent volume, so they are not lost when the app restarts.

Tech Stack

AI / Machine Learning

LLM Framework: langchain (for building the RAG pipeline)

LLM Server: ollama

Chat Model: mistral (7B)

Embedding Model: nomic-embed-text

Vector Database: chromadb (for storing document embeddings)

Document Loader: pypdf

DevOps & Backend

Backend API: fastapi

API Server: uvicorn

Containerization: docker & docker compose

Base Image: python:3.11-slim

How to Run This Project

There are two ways to run this application:

Production (Recommended): Using docker compose. This is the easiest and most reliable way.

Local Development: Running the Python server directly.

1. Production Mode (with Docker Compose)

This method automatically builds the API container and runs it alongside an official Ollama container.

Prerequisites:

Docker Desktop installed and running.

Steps:

Clone this repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name


Build the services:
This command builds the rag-api image from the Dockerfile.

docker compose build


Start the services:
This starts the rag-api and ollama containers in the background.

docker compose up -d


Pull the LLM models (One-Time Step):
You need to tell the ollama container to download the models. They will be saved in a persistent volume (./ollama_models) so you only do this once.

# Pull the chat model
docker compose exec ollama ollama pull mistral

# Pull the embedding model
docker compose exec ollama ollama pull nomic-embed-text


You're ready!
The API is now running. Open your browser to the documentation page to interact with it:
http://localhost:8000/docs

2. Local Development Mode

This method is for when you are actively developing the Python code.

Prerequisites:

Python 3.10+ and a virtual environment (venv).

The Ollama desktop app installed and running.

Steps:

Clone the repository and set up the environment:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Download Ollama models:
Run these commands in your regular terminal (not the venv):

ollama pull mistral
ollama pull nomic-embed-text


Run the FastAPI server:
The main.py file will automatically connect to your local Ollama app.

uvicorn main:app --reload


You're ready!
Open your browser to the documentation page:
http://localhost:8000/docs

API Endpoints

Once the server is running, you can use the /docs page to test the endpoints.

POST /upload/

Action: Uploads a PDF, splits it into chunks, embeds it, and saves it to the persistent vector store.

Body: multipart/form-data (select your file).

Response:

{
  "status": "success",
  "filename": "your_paper.pdf",
  "message": "File ingested successfully into the vector store."
}


POST /chat/

Action: Asks a question to the most recently uploaded PDF. Loads the vector store from disk, retrieves relevant chunks, builds the prompt, and returns the LLM's answer.

Body: application/x-www-form-urlencoded

query (string, required): Your question.

mode (string, optional): The persona to use. One of default, scientific, or company_analyst. Defaults to default.

Response:

{
  "answer": "The main conclusion of the paper is...",
  "mode": "scientific"
}