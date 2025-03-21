import os
import torch
import fitz
import chromadb
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import AutoModel, AutoTokenizer, pipeline

# Initialize FastAPI
app = FastAPI()

# Load Hugging Face Embedding Model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_hf_embedding(text):
    """Generate embeddings using Hugging Face model with GPU if available."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(name="document_qa_collection")


# Function to Extract Text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Function to Split Text into Chunks
def split_text(text, chunk_size=500, chunk_overlap=25):
    """Split text into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap  # Overlap ensures better retrieval
    return chunks

# Function to Load and Store Local Documents
def load_local_documents(directory_path="./documents"):
    """Load text and PDF files from a local folder and store them in ChromaDB."""
    if not os.path.exists(directory_path):
        print("Directory '{directory_path}' does not exist!")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print("Skipping unsupported file: {filename}")
            continue

        # Split text into chunks
        chunks = split_text(text)

        # Generate embeddings & store in ChromaDB
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk{i+1}"
            embedding = get_hf_embedding(chunk)
            collection.upsert(ids=[chunk_id], documents=[chunk], embeddings=[embedding])

# Load Documents on Startup
load_local_documents("./documents")

# Query Documents
@app.get("/query/")
async def query_documents(question: str):
    """Retrieve relevant documents from ChromaDB based on a query."""
    query_embedding = get_hf_embedding(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    if not results["documents"]:
        return JSONResponse(content={"relevant_text": []})

    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return JSONResponse(content={"relevant_text": relevant_chunks})

# Load QA Model
qa_model = pipeline("text2text-generation", model="google/flan-t5-large")

@app.get("/answer/")
async def generate_response(question: str):
    """Generate an AI-generated answer from retrieved document chunks."""
    relevant_data = await query_documents(question)

    if isinstance(relevant_data, JSONResponse):
        relevant_data = relevant_data.body  # Extract JSON content
        relevant_data = relevant_data.decode("utf-8")  # Decode bytes to string
        relevant_data = eval(relevant_data)  # Convert string to dictionary

    if "relevant_text" not in relevant_data or not relevant_data["relevant_text"]:
        return JSONResponse(content={"answer": "No relevant information found."})

    relevant_chunks = relevant_data["relevant_text"]
    context = "\n\n".join(relevant_chunks)

    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}"
    response = qa_model(prompt, max_length=150)

    return JSONResponse(content={"answer": response[0]['generated_text']})

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
