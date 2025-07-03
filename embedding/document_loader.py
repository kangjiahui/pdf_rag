import fitz
import json
import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import VECTOR_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def extract_toc(pdf_path: str) -> list:
    """提取并保存 TOC"""
    try:
        doc = fitz.open(pdf_path)
        toc_path = os.path.join(VECTOR_DIR, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_toc.json")
        if os.path.exists(toc_path):
            print(f"TOC 已存在，直接加载: {toc_path}")
            with open(toc_path, "r") as f:
                return json.load(f)
        else:
            print(f"正在提取 {pdf_path} 的目录...")
            toc = doc.get_toc()
            if not toc:
                print("⚠️ 未检测到目录")
                return []

        toc_data = [{"level": level, "title": title, "page": page - 1} for level, title, page in toc]

        os.makedirs(VECTOR_DIR, exist_ok=True)
        toc_path = os.path.join(VECTOR_DIR, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_toc.json")
        with open(toc_path, "w", encoding="utf-8") as f:
            json.dump(toc_data, f, ensure_ascii=False, indent=2)

        print(f"TOC 已保存至 {toc_path}")
        return toc_data

    except Exception as e:
        print(f"⚠️ 目录读取失败: {e}")
        return []
    
def build_chapter_page_ranges(toc: List[Dict], max_page: int) -> List[Dict]:
    """直接按 TOC 顺序切分，不筛叶子节点"""
    chapters = []
    for i, entry in enumerate(toc):
        start = entry["page"]
        end = toc[i + 1]["page"] if i + 1 < len(toc) else max_page
        chapters.append({
            "title": entry["title"],
            "start": start,
            "end": end
        })
        print(f"章节：{entry['title']} ({start} - {end})")
    return chapters

def load_pdf_by_page(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    return loader.load_and_split()

def split_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.create_documents([text])