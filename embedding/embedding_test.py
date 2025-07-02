from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer("BAAI/bge-large-zh")
texts = ["这是一个测试句子，用来生成向量。"] * 100

start = time.time()
embeddings = model.encode(texts)
print("生成完成，耗时：", time.time() - start)
print("每个向量维度：", len(embeddings[0]))