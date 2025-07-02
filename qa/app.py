from flask import Flask, request, jsonify
from document_loader import load_and_split
from embed_and_index import embed_documents, load_vector_store
from rag_qa import call_glm_api

app = Flask(__name__)

@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json()
    file_path = data.get("file_path")
    if not file_path:
        return jsonify({"error": "Missing file_path"}), 400

    docs = load_and_split(file_path)
    embed_documents(docs)
    return jsonify({"message": "Embedding completed."})

@app.route("/ask", methods=["GET"])
def ask():
    question = request.args.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400

    vs = load_vector_store()
    docs = vs.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"请根据以下内容回答问题：\n\n{context}\n\n问题：{question}"
    answer = call_glm_api(prompt)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
