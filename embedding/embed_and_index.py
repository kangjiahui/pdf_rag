# app/embed_and_index.py
import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from qa.config import VECTOR_DIR, PROGRESS_PATH
import pickle

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_progress(file_path, page_num):
    progress = load_progress()
    if file_path not in progress:
        progress[file_path] = []
    if page_num not in progress[file_path]:
        progress[file_path].append(page_num)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)

def is_page_done(file_path, page_num):
    progress = load_progress()
    return file_path in progress and page_num in progress[file_path]

def init_or_load_index():
    if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        return FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)
    else:
        return None