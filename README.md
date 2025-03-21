
This project is a lightweight **Retrieval-Augmented Generation (RAG)** system that uses a **FastAPI-based application** to allow users to query and generate answers from a collection of documents stored in **ChromaDB**. It leverages **Hugging Face's `sentence-transformers`** for generating embeddings and **`google/flan-t5-large`** for generating answers.

## Features

- **Document Ingestion**: Load and store text and PDF documents from a local directory.
- **Text Embedding**: Generate embeddings using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model.
- **Document Querying**: Retrieve relevant document chunks based on a user query.
- **Answer Generation**: Generate answers using the `google/flan-t5-large` model.
