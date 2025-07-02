from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import pickle
from config import VECTOR_DIR

# 初始化 embedding 模型
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")

def embed_documents_streaming(docs, batch_size=20):
    os.makedirs(VECTOR_DIR, exist_ok=True)

    index = None  # 向量库初始化为空
    doc_store = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        print(f"⏳ 正在嵌入第 {i} ~ {i+len(batch)-1} 条...")

        if index is None:
            # 初始化 index
            index = FAISS.from_documents(batch, embedding)
        else:
            # 增量添加
            index.add_documents(batch)

        doc_store.extend(batch)

    print("向量保存中...")
    index.save_local(VECTOR_DIR)

    with open(os.path.join(VECTOR_DIR, "doc_store.pkl"), "wb") as f:
        pickle.dump(doc_store, f)

    print("嵌入完成，已保存向量库。")