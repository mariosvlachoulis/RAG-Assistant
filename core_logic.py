import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# To run this file:
# 1. Make sure Ollama is running (e.g., `ollama serve`)
# 2. Make sure you have pulled the models:
#    `ollama pull mistral`
#    `ollama pull nomic-embed-text`
# 3. Place a PDF in this directory (e.g., `test_paper.pdf`)
# 4. Run from your terminal:
#    `python core_logic.py --pdf "test_paper.pdf" --query "What is this paper about?" --mode "default"`
# ----------------

# --- PERSONA PROMPT TEMPLATES ---
# We define our different "persona" prompts here.

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

# We map the string 'mode' to the actual prompt template
PROMPT_TEMPLATES = {
    "default": PromptTemplate.from_template(DEFAULT_PERSONA_PROMPT),
    "scientific": PromptTemplate.from_template(SCIENTIFIC_PERSONA_PROMPT),
    "company_analyst": PromptTemplate.from_template(COMPANY_ANALYST_PERSONA_PROMPT),
}

# --- 1. DOCUMENT INGESTION ---

def load_and_split_pdf(pdf_path):
    """
    Loads a PDF, splits it into manageable chunks.
    This is the "L" and "S" in RAG (Load, Split).
    """
    print(f"Loading and splitting PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Successfully split PDF into {len(chunks)} chunks.")
    return chunks

# --- 2. EMBEDDING & VECTOR STORE ---

def get_vector_store(chunks):
    """
    Embeds the chunks and stores them in a Chroma vector database.
    This is the "E" and "S" in RAG (Embed, Store).
    
    We are using a "transient" in-memory vector store for this script.
    In Step 2 (FastAPI), we will make this persistent.
    """
    print("Initializing embedding model and vector store...")
    # Initialize the embedding model we'll use
    # "nomic-embed-text" is a great open-source model.
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Create the vector store from the document chunks
    # This will embed all chunks and store them.
    vector_store = Chroma.from_documents(
        chunks,
        embedding_model,
        collection_name="rag-collection",
        persist_directory="./chroma_db_local" # We'll persist to disk for later steps
    )
    print("Vector store created successfully.")
    return vector_store

def get_retriever(pdf_path):
    """
    A helper function to combine ingestion and vector store creation.
    It will load the PDF, create the vector store, and return a "retriever"
    object that we can use to find relevant documents.
    """
    chunks = load_and_split_pdf(pdf_path)
    vector_store = get_vector_store(chunks)
    
    # A retriever is a component that "retrieves" documents
    # k=3 means it will find the top 3 most relevant chunks.
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    print("Retriever is ready.")
    return retriever


# --- 3. RAG CHAIN (GENERATION) ---

def get_rag_chain(retriever, mode="default"):
    """
    Creates the main RAG "chain" using LangChain (LCEL).
    This chain will:
    1. Take a user's question.
    2. Find relevant documents (Retrieve).
    3. Format the prompt with the persona (Augment).
    4. Pass it to the LLM to get an answer (Generate).
    """
    print(f"Building RAG chain with '{mode}' persona...")
    
    # Initialize our LLM
    llm = OllamaLLM(model="mistral")

    # Select the correct prompt template based on the mode
    prompt = PROMPT_TEMPLATES.get(mode)
    if not prompt:
        print(f"Warning: Mode '{mode}' not found. Using 'default'.")
        prompt = PROMPT_TEMPLATES["default"]

    # This function will format the retrieved documents into a string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # This is the LangChain Expression Language (LCEL) chain
    # It defines the flow of data.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain built successfully.")
    return rag_chain

# --- 4. MAIN EXECUTION ---

def main(pdf_path, query, mode):
    """
    Main function to run the RAG pipeline as a script.
    """
    try:
        # 1. Create the retriever
        retriever = get_retriever(pdf_path)
        
        # 2. Create the RAG chain
        rag_chain = get_rag_chain(retriever, mode)
        
        # 3. Get the answer
        print("\n--- QUERY ---")
        print(f"Query: {query}")
        print(f"Mode: {mode}")
        print("\n--- ANSWER ---")
        
        # We use .stream() to get a streaming response, like ChatGPT
        # This is much better for user experience.
        chunks = []
        for chunk in rag_chain.stream(query):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        
        print("\n\n--- END OF RESPONSE ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    # This allows us to run the script from the command line
    parser = argparse.ArgumentParser(description="Run a RAG query on a PDF.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file.")
    parser.add_argument("--query", type=str, required=True, help="The question to ask the PDF.")
    parser.add_argument("--mode", type=str, default="default",
                        choices=PROMPT_TEMPLATES.keys(),
                        help="The persona mode to use.")
    
    args = parser.parse_args()
    
    # Make sure Ollama is running!
    main(args.pdf, args.query, args.mode)