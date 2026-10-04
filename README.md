# AI-Based Panchayat Record Assistant (OCR + RAG)

An AI-assisted document processing application for digitizing Panchayat records and answering questions from uploaded documents.

## Features

- Upload scanned Panchayat record images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`)
- OCR using Tesseract
- OpenCV image preprocessing for better OCR
- English/Tamil OCR language support
- Text chunking and semantic retrieval
- Sentence Transformers embeddings
- FAISS vector search
- RAG-based question answering
- Optional Groq LLM integration for natural-language answers
- Flask REST API
- Browser-based dashboard
- Extracted text preview
- Source chunk display

## Project Structure

```text
AI-Based-Panchayat-Record-Assistant/
├── backend/
│   ├── app.py
│   ├── ocr.py
│   ├── rag.py
│   ├── requirements.txt
│   └── uploads/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── .env.example
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- Tesseract OCR
- Internet connection for the first download of the Sentence Transformer model

### Install Tesseract on Windows

Install Tesseract OCR and make sure the executable path is available.

Typical path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

For Tamil OCR, install the Tamil `tam.traineddata` file in the Tesseract `tessdata` folder.

If Tesseract is not in PATH, edit `backend/ocr.py` and set:

```python
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Setup

Open the project folder in VS Code.

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate it

Windows:

```bash
venv\Scripts\activate
```

### 3. Install packages

```bash
pip install -r backend/requirements.txt
```

### 4. Optional: configure Groq

Copy `.env.example` to `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

The application still works without the key using a retrieval-based fallback answer.

### 5. Run

```bash
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## API Endpoints

### Upload document

```http
POST /upload
Content-Type: multipart/form-data
file=<image>
```

### Ask a question

```http
POST /query
Content-Type: application/json

{
  "question": "Who is the owner mentioned in the document?"
}
```

### Health check

```http
GET /health
```

## RAG Workflow

```text
Scanned Panchayat Record
        |
        v
 OpenCV Preprocessing
        |
        v
   Tesseract OCR
        |
        v
 Extracted Text
        |
        v
 Text Chunking
        |
        v
 Sentence Transformer
        |
        v
    FAISS Index
        |
        v
 Relevant Chunks
        |
        v
   Groq LLM / Fallback
        |
        v
 Context-Aware Answer
```

## Resume Technology Stack

Python, Tesseract OCR, OpenCV, FAISS, Flask, RAG, LLMs, Sentence Transformers
