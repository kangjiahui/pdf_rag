GLM_API_KEY = "xxx"  # 替换为你的智谱 API Key

VECTOR_DIR = "./qa/vector_store"
PROGRESS_PATH = "./embedding/progress.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# 动态 Top-K 参数
MAX_TOP_K = 8
SCORE_THRESHOLD = 0.3  # 距离越小越相似，按 faiss score 取