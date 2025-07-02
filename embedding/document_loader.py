# app/document_loader.py
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qa.config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf_by_page(file_path):
    doc = fitz.open(file_path)
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text()
        yield i, text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.create_documents([text])

def get_tail(text, max_chars=200):
    return text[-max_chars:] if len(text) > max_chars else text