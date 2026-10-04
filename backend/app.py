import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ocr import extract_text
from rag import build_index, retrieve, generate_answer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

DOCUMENT_STORE = {
    "filename": None,
    "text": "",
    "chunk_count": 0,
}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.get("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "document_loaded": bool(DOCUMENT_STORE["text"]),
        "filename": DOCUMENT_STORE["filename"],
    })


@app.post("/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file was selected."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file type. Upload PNG, JPG, JPEG, TIF or TIFF."
        }), 400

    filename = secure_filename(file.filename)
    filepath = UPLOAD_DIR / filename
    file.save(filepath)

    try:
        languages = request.form.get("languages", "eng+tam")
        text = extract_text(str(filepath), languages)
    except Exception as exc:
        return jsonify({
            "error": (
                "OCR failed. Make sure Tesseract OCR is installed and the "
                "required language data is available."
            ),
            "details": str(exc),
        }), 500

    if not text:
        return jsonify({
            "error": "No readable text was detected in the document."
        }), 400

    chunk_count = build_index(text)

    DOCUMENT_STORE["filename"] = filename
    DOCUMENT_STORE["text"] = text
    DOCUMENT_STORE["chunk_count"] = chunk_count

    return jsonify({
        "message": "Document uploaded and indexed successfully.",
        "filename": filename,
        "text": text,
        "chunk_count": chunk_count,
    })


@app.post("/query")
def query():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Enter a question."}), 400

    if not DOCUMENT_STORE["text"]:
        return jsonify({"error": "Upload a Panchayat record first."}), 400

    contexts = retrieve(question, top_k=4)
    answer = generate_answer(question, contexts)

    return jsonify({
        "question": question,
        "answer": answer,
        "sources": contexts,
    })


@app.get("/document")
def document():
    return jsonify({
        "filename": DOCUMENT_STORE["filename"],
        "text": DOCUMENT_STORE["text"],
        "chunk_count": DOCUMENT_STORE["chunk_count"],
    })


if __name__ == "__main__":
    print("AI-Based Panchayat Record Assistant")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=True)
