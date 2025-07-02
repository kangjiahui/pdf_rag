# build_embedding.py
from document_loader import load_and_split
from embed_and_index import embed_documents_streaming

# 修改为你放入的文档路径
file_path = "docs/Matter Specification 1.4.1.pdf"

docs = load_and_split(file_path)
embed_documents_streaming(docs, batch_size=10)  # 可调节 batch_size
print("文档嵌入构建完成并已保存本地。")