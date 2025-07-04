import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from embedding.document_loader import extract_toc, build_chapter_page_ranges, load_pdf_by_page, split_text
from config import VECTOR_DIR, PROGRESS_PATH

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_progress(file_path, title):
    progress = load_progress()
    if file_path not in progress:
        progress[file_path] = []
    if title not in progress[file_path]:
        progress[file_path].append(title)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)

def build_or_load_index():
    if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        print("加载已有向量库...")
        return FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)
    else:
        print("尚未创建向量库")
        return None

def save_index(index):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    index.save_local(VECTOR_DIR)
    print(f"向量库已保存至 {VECTOR_DIR}")

def estimate_chunk_pages(chunks, pages):
    """估算每个 chunk 覆盖的页码范围"""
    total_chars = sum(len(text) for _, text in pages)
    pages_list = [p for p, _ in pages]

    # 按比例粗略估算页码范围
    chunk_ranges = []
    char_accum = 0
    page_idx = 0

    for chunk in chunks:
        chunk_chars = len(chunk.page_content)
        char_accum += chunk_chars
        ratio = char_accum / total_chars
        approx_page = int(ratio * (pages_list[-1] - pages_list[0] + 1)) + pages_list[0]
        chunk_ranges.append((pages_list[0], min(approx_page, pages_list[-1])))
        # 下一 chunk 从当前页码开始
        pages_list[0] = min(approx_page, pages_list[-1])

    # 保证页码不倒序
    ranges = []
    for start, end in chunk_ranges:
        if end < start:
            end = start
        ranges.append((start, end))
    return ranges

def process_pdf_streaming(pdf_path: str):
    print(f"正在处理 PDF 文件：{pdf_path}")
    filename = os.path.basename(pdf_path)
    docs = load_pdf_by_page(pdf_path)
    max_page = max(doc.metadata.get("page", 0) for doc in docs)
    page_docs = {doc.metadata["page"]: doc for doc in docs}

    toc = extract_toc(pdf_path)
    index = build_or_load_index()

    if toc:
        chapters = build_chapter_page_ranges(toc, max_page)
        print(f"章节数：{len(chapters)}")

        for chapter in chapters:
            # 拼接章节所有页的文本
            pages = [(p, page_docs[p].page_content) for p in range(chapter["start"], chapter["end"] + 1) if p in page_docs]
            full_text = "\n".join([t for _, t in pages])

            # 如果章节不大，直接 embedding
            if len(full_text) <= 1000:
                meta = {
                    "source": filename,
                    "chapter": chapter["title"],
                    "start_page": chapter["start"],
                    "end_page": chapter["end"],
                    "chunk_start_page": chapter["start"],
                    "chunk_end_page": chapter["end"],
                }
                doc = Document(page_content=full_text, metadata=meta)
                if index is None:
                    index = FAISS.from_documents([doc], embedding)
                else:
                    index.add_documents([doc])
                save_index(index)
                print(f"已处理章节：{chapter['title']}")
            else:
                # 章节过大，切为多个 chunk
                chunks = split_text(full_text)
                chunk_page_ranges = estimate_chunk_pages(chunks, pages)

                for chunk, (chunk_start, chunk_end) in zip(chunks, chunk_page_ranges):
                    meta = {
                        "source": filename,
                        "chapter": chapter["title"],
                        "start_page": chapter["start"],
                        "end_page": chapter["end"],
                        "chunk_start_page": chunk_start,
                        "chunk_end_page": chunk_end,
                    }
                    chunk.metadata.update(meta)

                    if index is None:
                        index = FAISS.from_documents([chunk], embedding)
                    else:
                        index.add_documents([chunk])
                    save_index(index)
                print(f"已处理大章节：{chapter['title']}，共 {len(chunks)} 个 chunk")

    else:
        print("⚠️ 无目录，按页切分")
        for doc in docs:
            meta = {
                "source": filename,
                "chapter": f"第{doc.metadata.get('page', -1)}页",
                "start_page": doc.metadata.get('page', -1),
                "end_page": doc.metadata.get('page', -1),
                "chunk_start_page": doc.metadata.get('page', -1),
                "chunk_end_page": doc.metadata.get('page', -1),
            }
            chunks = split_text(doc.page_content)
            for chunk in chunks:
                chunk.metadata.update(meta)
                if index is None:
                    index = FAISS.from_documents([chunk], embedding)
                else:
                    index.add_documents([chunk])
                save_index(index)
            print(f"已处理第 {doc.metadata.get('page', -1)} 页")

    print("完成")

if __name__ == "__main__":
    # pdf_path = input("请输入 PDF 文件路径：").strip()
    # pdf_path = "docs/Matter Specification 1.4.1.pdf"
    pdf_path = "docs/Aliro_v0.9.0.pdf"
    process_pdf_streaming(pdf_path)