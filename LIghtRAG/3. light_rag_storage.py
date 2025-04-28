# storage.py
import os
from lightrag import LightRAG
from embedder import get_text_embedding

WORKING_DIR = "./rag_storage"

async def initialize_light_rag():
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=get_text_embedding,
        llm_model_func=lambda prompt: "Placeholder LLM call",
    )
    await rag.initialize_storages()
    return rag

async def insert_documents(rag, documents):
    for doc in documents:
        await rag.insert(doc)
