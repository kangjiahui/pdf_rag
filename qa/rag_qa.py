import os
import requests
from config import GLM_API_KEY
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from config import VECTOR_DIR, MAX_TOP_K, SCORE_THRESHOLD

# ----------- 配置区 -----------
ZHIPU_CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4"

# ----------- 向量库加载 -----------
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
db = FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)

# ----------- 保存历史消息 -----------
message_history = []

# ----------- 问答函数 -----------
def query_rag(question: str, top_k: int = 5):
    # 1. 检索文档相关内容
    docs = db.similarity_search(question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. 加入 context 到 messages
    # system + 所有历史 + 当前用户问题
    messages = [{"role": "system", "content": "你是一个行业规范解读专家，请根据用户提问和相关文档内容作答。"}]
    messages.extend(message_history)  # 添加历史
    messages.append({
        "role": "user",
        "content": f"相关文档如下：\n{context}\n\n问题是：{question}"
    })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GLM_API_KEY}"
    }

    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2
    }

    try:
        resp = requests.post(ZHIPU_CHAT_URL, json=body, headers=headers)
        res = resp.json()

        if "choices" in res:
            reply = res["choices"][0]["message"]["content"]
            # 将本轮问答加入历史
            message_history.append({"role": "user", "content": question})
            message_history.append({"role": "assistant", "content": reply})
            return reply
        else:
            print("返回结构异常:", res)
            return "⚠️ 接口返回异常。"

    except Exception as e:
        print("请求失败:", e)
        return "⚠️ 无法连接大模型。"

# ---- CLI 交互 ----
if __name__ == "__main__":
    print("智谱 RAG 问答系统，支持多轮对话（输入 q 退出）")
    while True:
        query = input("\n请输入问题：")
        if query.strip().lower() in ["q", "quit", "exit"]:
            print("退出。")
            break
        answer = query_rag(query)
        print("\nchatGLM 回答：\n", answer)
