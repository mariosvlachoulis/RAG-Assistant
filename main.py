import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever


# To run this file:
# 1. Make sure Ollama is running (e.g., `ollama serve`)
# 2. Make sure you have pulled the models:
#    `ollama pull mistral`
#    `ollama pull nomic-embed-text`
# 3. Run the API server from your terminal:
#    `uvicorn main:app --reload`
# 4. Open your browser to `http://127.0.0.1:8000/docs`
# ----------------

# --- Constants ---
# We'll create a *new* persistent directory for the API
CHROMA_PERSIST_DIR = "./chroma_db_api"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="An API for chatting with your PDFs using local LLMs."
)

# --- PERSONA PROMPT TEMPLATES ---

DEFAULT_PERSONA_PROMPT = """
You are a helpful assistant. Answer the user's question based *only* on the
following context. If the context does not contain the answer, state
that you are unable to answer.

Context:
{context}

Question:
{question}
"""

SCIENTIFIC_PERSONA_PROMPT = """
You are a meticulous research assistant. Your task is to provide a precise,
formal, and factual answer to the user's question, based *strictly* on the
provided context.

- Answer *only* using the information from the text.
- Be formal and objective.
- If the answer is not in the context, state: "The provided text does not
  contain information on this topic."
- **Crucially:** Do not add any external information, interpretation, or opinion.

Context:
{context}

Question:
{question}
"""

COMPANY_ANALYST_PERSONA_PROMPT = """
You are a savvy business analyst and career coach. Your goal is to help a
job applicant understand a company. Analyze the provided text (e.g., job
description, company website 'About' page) to answer the user's question.

- Focus on insights related to company culture, mission, vision, and values.
- Adopt an encouraging, insightful, and professional tone.
- If the text is a job listing, you can infer what it suggests about the company.
- If the answer is not in the context, state that the information is not provided.

Context:
{context}

Question:
{question}
"""

PROMPT_TEMPLATES = {
    "default": PromptTemplate.from_template(DEFAULT_PERSONA_PROMPT),
    "scientific": PromptTemplate.from_template(SCIENTIFIC_PERSONA_PROMPT),
    "company_analyst": PromptTemplate.from_template(COMPANY_ANALYST_PERSONA_PROMPT),
}

# --- Core RAG Logic (Refactored for API) ---

def ingest_pdf(pdf_path: str) -> bool:
    """
    Loads, splits, embeds, and stores a PDF in the persistent vector store.
    """
    try:
        print(f"Starting ingestion for: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Split PDF into {len(chunks)} chunks.")

        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # Create the vector store with persistence
        vector_store = Chroma.from_documents(
            chunks,
            embedding_model,
            collection_name="rag-collection-api",
            persist_directory=CHROMA_PERSIST_DIR
        )
        vector_store.persist()
        print(f"Successfully ingested and persisted vectors to {CHROMA_PERSIST_DIR}")
        return True
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return False

def get_persistent_retriever() -> BaseRetriever:
    """
    Loads the persistent vector store and returns a retriever.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise HTTPException(status_code=404, detail="Vector store not found. Please upload a PDF first.")
        
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    vector_store = Chroma(
        collection_name="rag-collection-api",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model
    )
    
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    return retriever

def get_rag_chain(retriever: BaseRetriever, mode: str):
    """
    Creates the RAG chain using the provided retriever and mode.
    """
    llm = Ollama(model=LLM_MODEL)
    
    prompt = PROMPT_TEMPLATES.get(mode)
    if not prompt:
        print(f"Warning: Mode '{mode}' not found. Using 'default'.")
        prompt = PROMPT_TEMPLATES["default"]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- API Endpoints ---

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file.
    It saves the file temporarily, ingests it into the vector store,
    and then deletes the temporary file.
    """
    # Use a temporary file to save the uploaded PDF
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        print(f"PDF saved temporarily to: {tmp_path}")
        
        # Ingest the PDF
        success = ingest_pdf(tmp_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to ingest PDF.")

        return {
            "status": "success",
            "filename": file.filename,
            "message": "File ingested successfully into the vector store."
        }
    except Exception as e:
        # Handle potential exceptions
        print(f"Error in /upload/: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Ensure the temporary file is deleted
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"Temporary file {tmp_path} deleted.")

@app.post("/chat/")
async def chat_with_pdf(
    query: str = Form(...),
    mode: str = Form("default")
):
    """
    Endpoint to chat with the ingested PDF.
    It loads the persistent vector store, creates the RAG chain,
    and returns the LLM's answer.
    """
    try:
        # 1. Load the retriever
        retriever = get_persistent_retriever()
        
        # 2. Create the RAG chain
        rag_chain = get_rag_chain(retriever, mode)
        
        # 3. Get the answer (synchronous for simplicity)
        # We use .invoke() here for a simple request/response.
        answer = rag_chain.invoke(query)
        
        return {
            "answer": answer,
            "mode": mode
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions (like 404)
        raise e
    except Exception as e:
        # Handle other runtime errors
        print(f"Error in /chat/: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Uvicorn Server ---
if __name__ == "__main__":
    import uvicorn
    # This allows you to run the app by executing `python main.py`
    # --reload is great for development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)