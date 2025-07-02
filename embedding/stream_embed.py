# stream_embed.py
import os
from document_loader import load_pdf_by_page, split_text, get_tail
from embed_and_index import (
    embedding,
    init_or_load_index,
    is_page_done,
    save_progress
)
from qa.config import VECTOR_DIR, TAIL_OVERLAP_CHARS

from langchain.vectorstores import FAISS

file_path = "docs/Matter Specification 1.4.1.pdf"

os.makedirs(VECTOR_DIR, exist_ok=True)
index = init_or_load_index()
prev_tail = ""

for page_num, page_text in load_pdf_by_page(file_path):
    if is_page_done(file_path, page_num):
        print(f"已处理页 {page_num}")
        prev_tail = get_tail(page_text, TAIL_OVERLAP_CHARS)
        continue

    print(f"正在处理页 {page_num}")
    combined_text = prev_tail + "\n" + page_text
    chunks = split_text(combined_text)

    if not chunks:
        continue

    if index is None:
        index = FAISS.from_documents(chunks, embedding)
    else:
        index.add_documents(chunks)

    index.save_local(VECTOR_DIR)
    save_progress(file_path, page_num)
    prev_tail = get_tail(combined_text, TAIL_OVERLAP_CHARS)

print("所有页面处理完毕，向量库已完成构建！")