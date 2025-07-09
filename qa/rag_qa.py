import os
import requests
from config import GLM_API_KEY
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import VECTOR_DIR, MAX_TOP_K, SCORE_THRESHOLD

def load_index():
    if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        raise FileNotFoundError("未找到向量库 index.faiss")
    return FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)

def search_index(index, query):
    docs_and_scores = index.similarity_search_with_score(query, k=MAX_TOP_K * 2)
    filtered = [(doc, score) for doc, score in docs_and_scores if score >= SCORE_THRESHOLD]
    filtered = sorted(filtered, key=lambda x: -x[1])  # 按得分降序排列
    top_k = filtered[:MAX_TOP_K]
    return top_k

def build_prompt(query, docs):
    context_text = ""
    references = []
    for i, (doc, score) in enumerate(docs):
        metadata = doc.metadata
        context_text += f"[文档{i+1}] {doc.page_content}\n"
        references.append(metadata)
    prompt = f"以下是规范文档内容，请根据这些内容回答问题。\n\n{context_text}\n\n问题：{query}\n\n请基于文档回答，不要编造。\n"
    return prompt, references

# ----------- 配置区 -----------
ZHIPU_CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4"
TOKENIZER_MODE = "api" # 调用glm-api使用"api"; 本地部署Qwen使用 "local"

# ----------- 向量库加载 -----------
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
db = FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)

# ----------- 本地模型加载 -----------
if (TOKENIZER_MODE == "local"):
    # 首次运行会自动下载 Qwen1.5-1.8B-Chat
    LOCAL_MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True).eval()  # 默认 CPU

    # 使用 pipeline 简化调用
    chat_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # -1 表示 CPU，M1 Air 上默认即可
    )

# ----------- 问答函数 -----------
def query_rag(query, chat_history):
    # 1. 检索文档相关内容
    index = load_index()
    top_docs = search_index(index, query)
    prompt, references = build_prompt(query, top_docs)

    # 2. 加入 context 到 messages
    # system + 所有历史 + 当前用户问题
    if (TOKENIZER_MODE == "api"):
        messages = chat_history + [{"role": "user", "content": prompt}]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GLM_API_KEY}"
        }

        body = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.2
        }
    else:
        history_text = ""
        for message in chat_history:
            if message["role"] == "user":
                history_text += f"用户：{message['content']}\n"
            elif message["role"] == "assistant":
                history_text += f"助手：{message['content']}\n"

        # 3. 拼接完整 prompt，符合 Qwen 对话格式
        full_prompt = history_text + f"用户：{prompt}\n助手："


    try:
        if (TOKENIZER_MODE == "api"):
            resp = requests.post(ZHIPU_CHAT_URL, json=body, headers=headers)
            res = resp.json()
            if "choices" in res:
                reply = res["choices"][0]["message"]["content"]
                # 将本轮问答加入历史
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": reply})
                # ref_text = "\n\n📎 参考来源：\n" + "\n".join(references)
                return reply, references
            else:
                print("返回结构异常:", res)
                return "接口返回异常。", ""
        
        else:
            outputs = chat_pipeline(
                full_prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

            # 去掉 prompt 部分，保留模型回复
            reply = outputs[0]['generated_text'][len(full_prompt):].strip()

            # 5. 更新 chat_history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": reply})
        
            return reply, references 

    except Exception as e:
        print("模型推理失败:", e)
        return "无法连接大模型。", ""

# ---- CLI 交互 ----
if __name__ == "__main__":
    # ----------- 保存历史消息 -----------
    message_history = []
    print("智谱 RAG 问答系统，支持多轮对话（输入 q 退出）")
    while True:
        query = input("\n请输入问题：")
        if query.strip().lower() in ["q", "quit", "exit"]:
            print("退出。")
            break
        answer = query_rag(query, message_history)
        print("\nchatGLM 回答：\n", answer)
