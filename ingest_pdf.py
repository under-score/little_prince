#!/usr/bin/env python3
# ============================================================
#  PDF indexer for LM Studio
#  Author: chat GPT 5.0
#  Date:   2025-11-12
# ============================================================

import os
from tqdm import tqdm
from multiprocessing import get_context, TimeoutError
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === CONFIG ===
PDF_DIR = "⁓/legal"
INDEX_BASE_DIR = "⁓/AIIndexStore"
INDEX_BASE_DIR = "⁓/AIIndexStore"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_API_BASE = "http://localhost:11434/v1"
CHUNK_SIZE = 10000
START_CHUNK = 1
END_CHUNK = 1
BAD_LOG = os.path.join(INDEX_BASE_DIR, "bad_files.txt")
TIMEOUT_SECONDS = 60

# === Step 1: Get all PDFs ===
def get_all_pdfs(path):
    pdf_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return sorted(pdf_files)

# === Step 2: Timeout-safe PDF parsing ===
def parse_pdf_safe(path):
    try:
        text = extract_text(path)
        return {"success": True, "doc": Document(text=text, metadata={"file_path": path})}
    except Exception as e:
        return {"success": False, "file_path": path, "error": str(e)}

def parse_pdfs_parallel(paths, timeout=TIMEOUT_SECONDS):
    documents = []
    print(f"Parsing {len(paths)} PDFs (timeout: {timeout}s each)...")

    ctx = get_context("spawn")
    pool = ctx.Pool(processes=os.cpu_count())

    try:
        for path in tqdm(paths, desc="Parsing PDFs"):
            async_result = pool.apply_async(parse_pdf_safe, (path,))
            try:
                result = async_result.get(timeout=timeout)
                if result["success"]:
                    documents.append(result["doc"])
                else:
                    print(f"{result['file_path']}: {result['error']}")
                    with open(BAD_LOG, "a") as log:
                        log.write(f"FAIL: {result['file_path']} :: {result['error']}\n")
            except TimeoutError:
                print(f"⏰ Timeout: {path}")
                with open(BAD_LOG, "a") as log:
                    log.write(f"TIMEOUT: {path}\n")
    finally:
        pool.terminate()
        pool.join()

    return documents

# === Step 3: Build and save index ===
def build_index(documents, index_dir):
    print(f"🔍 Indexing {len(documents)} documents → {index_dir}")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # llm = OpenAI(api_base="http://localhost:11434/v1", api_key="lm-studio")
    llm = Ollama(model="llama3")

    Settings.embed_model = embed_model
    Settings.llm = llm

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_dir)
    print(f"Index saved to {index_dir}")

# === MAIN ===
if __name__ == "__main__":
    all_pdfs = get_all_pdfs(PDF_DIR)
    total = len(all_pdfs)
    total_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE

    os.makedirs(INDEX_BASE_DIR, exist_ok=True)
    open(BAD_LOG, "w").close()  # clear previous log

    for chunk_num in range(START_CHUNK, END_CHUNK + 1):
        start = (chunk_num - 1) * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total)
        chunk_paths = all_pdfs[start:end]

        index_dir = os.path.join(INDEX_BASE_DIR, f"part_{chunk_num}")
        if os.path.exists(index_dir):
            print(f"Chunk {chunk_num} already exists, skipping: {index_dir}")
            continue

        print(f"\nProcessing chunk {chunk_num}/{total_chunks} → files {start+1} to {end}")
        docs = parse_pdfs_parallel(chunk_paths)
        build_index(docs, index_dir)
