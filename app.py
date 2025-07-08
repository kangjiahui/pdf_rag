import re
import os
from flask import Flask, request, jsonify, render_template
from markupsafe import Markup
from embedding.stream_embed import process_pdf_streaming
from qa.rag_qa import query_rag

UPLOAD_FOLDER = "docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_answer(text):
    # 去掉 markdown 粗体、标题等
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # **加粗**
    text = re.sub(r"#\s*", "", text)  # # 标题
    return text

app = Flask(__name__)
chat_history = []  # 简单内存记录，可持久化

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # 上传成功后调用 embed
    return jsonify({"status": "success", "pdf_path": save_path})

@app.route("/embed", methods=["POST"])
def embed_pdf():
    pdf_path = request.json.get("pdf_path")
    if not pdf_path:
        return jsonify({"error": "Missing 'pdf_path' parameter"}), 400
    try:
        process_pdf_streaming(pdf_path)
        return jsonify({"status": "success", "message": f"Embedding completed for {pdf_path}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("query")
    global chat_history
    answer, sources = query_rag(query, chat_history)
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    # 清理 markdown、格式优化
    answer_clean = clean_answer(answer).replace("\n", "<br>")
    # sources 是 references 的列表
    source_info = "".join(
        f"<div class='text-secondary ms-3'>📎 [文档] {src.get('source', '')} | {src.get('chapter', '')} | 第 {src.get('start_page', '')} 页</div>"
        for src in sources
    )

    html = f"""
    <div class="mb-2"><b>你：</b>{query}</div>
    <div class="mb-2 text-primary"><b>AI：</b>{Markup(answer_clean)}</div>
    <div class="mb-2 text-secondary">{Markup(source_info)}</div>
    <hr>
    """
    return html



# 保留 API 接口
@app.route("/api/qa", methods=["POST"])
def rag_qa_api():
    query = request.json.get("query")
    history = request.json.get("history", [])
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    try:
        answer, source = query_rag(query, history)
        return jsonify({"status": "success", "answer": answer})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)