from __future__ import annotations

import os
import re
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from groq import Groq
except ImportError:
    Groq = None


MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(MODEL_NAME)

document_chunks: List[str] = []
index = None


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80) -> List[str]:
    """Split OCR text into overlapping chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_index(text: str) -> int:
    """Create a FAISS inner-product index for the current document."""
    global document_chunks, index

    document_chunks = chunk_text(text)

    if not document_chunks:
        index = None
        return 0

    embeddings = embedding_model.encode(
        document_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return len(document_chunks)


def retrieve(question: str, top_k: int = 4) -> List[Dict]:
    """Retrieve the most relevant chunks for a question."""
    if index is None or not document_chunks:
        return []

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, len(document_chunks))
    scores, ids = index.search(query_embedding, k)

    results = []

    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id == -1:
            continue

        results.append({
            "chunk": document_chunks[int(chunk_id)],
            "score": float(score),
        })

    return results


# ============================================================
# LOCAL EXACT ANSWER EXTRACTION
# ============================================================

def extract_exact_answer(question: str, context: str) -> str | None:
    """
    Extract a short exact answer from OCR text when Groq is unavailable.
    This prevents the application from returning the entire paragraph.
    """

    q = question.lower().strip()

    # --------------------------------------------------------
    # 1. REGISTRATION NUMBER
    # --------------------------------------------------------

    registration_keywords = [
        "registration number",
        "registration no",
        "registration",
        "பதிவு எண்",
        "பதிவு",
        "அரசு பதிவு எண்",
    ]

    if any(keyword in q for keyword in registration_keywords):

        # Example: 214/2025
        match = re.search(
            r"\b\d{1,6}\s*/\s*\d{4}\b",
            context
        )

        if match:
            return match.group(0).replace(" ", "")

    # --------------------------------------------------------
    # 2. DATE
    # --------------------------------------------------------

    date_keywords = [
        "date",
        "mentioned date",
        "document date",
        "what is the date",
        "தேதி",
        "நாள்",
    ]

    if any(keyword in q for keyword in date_keywords):

        # Matches:
        # 12.03.2025
        # 12/03/2025
        # 12-03-2025
        match = re.search(
            r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",
            context
        )

        if match:
            value = match.group(0)

            # Standardize separator
            value = value.replace("/", ".").replace("-", ".")

            return value

    # --------------------------------------------------------
    # 3. FINANCIAL YEAR
    # --------------------------------------------------------

    financial_year_keywords = [
        "financial year",
        "financial",
        "year",
        "ஆண்டு",
        "நிதியாண்டு",
    ]

    if any(keyword in q for keyword in financial_year_keywords):

        # Matches:
        # 2024 - 2025
        # 2024-2025
        # 2024 – 2025
        match = re.search(
            r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b",
            context
        )

        if match:
            return f"{match.group(1)}-{match.group(2)}"

    # --------------------------------------------------------
    # 4. AMOUNT / TAX AMOUNT
    # --------------------------------------------------------

    amount_keywords = [
        "amount",
        "tax amount",
        "total amount",
        "மொத்த தொகை",
        "தொகை",
        "வரி தொகை",
    ]

    if any(keyword in q for keyword in amount_keywords):

        # Currency amounts such as:
        # ₹500
        # Rs. 500
        # ரூ.500
        match = re.search(
            r"(?:₹|Rs\.?|ரூ\.?)\s*[\d,]+(?:\.\d+)?",
            context,
            re.IGNORECASE
        )

        if match:
            return match.group(0).strip()

    # --------------------------------------------------------
    # 5. SURVEY NUMBER
    # --------------------------------------------------------

    survey_keywords = [
        "survey number",
        "survey no",
        "survey",
        "சர்வே எண்",
        "சர்வே",
    ]

    if any(keyword in q for keyword in survey_keywords):

        match = re.search(
            r"(?:survey\s*(?:no|number)?|சர்வே\s*எண்?)\s*[:\-]?\s*([0-9A-Za-z./-]+)",
            context,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

    # --------------------------------------------------------
    # 6. FALLBACK — RETURN ONE RELEVANT SENTENCE
    # --------------------------------------------------------

    # Instead of returning the whole document,
    # return only a short relevant sentence.

    sentences = re.split(r"(?<=[.!?])\s+|\n+", context)

    question_words = set(
        re.findall(r"[a-zA-Z]{3,}", q)
    )

    best_sentence = None
    best_score = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = set(
            re.findall(r"[a-zA-Z]{3,}", sentence.lower())
        )

        score = len(question_words.intersection(sentence_words))

        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_sentence:
        return best_sentence

    # Last resort: return only the first short portion
    words = context.split()

    if words:
        return " ".join(words[:30])

    return None


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(question: str, contexts: List[Dict]) -> str:
    """
    Generate an answer using Groq if configured.
    Otherwise extract an exact short answer locally.
    """

    if not contexts:
        return "No relevant information was found."

    context_text = "\n\n".join(
        f"[Source {i + 1}]\n{item['chunk']}"
        for i, item in enumerate(contexts)
    )

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    # --------------------------------------------------------
    # GROQ LLM
    # --------------------------------------------------------

    if api_key and Groq is not None:

        client = Groq(api_key=api_key)

        model = os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant"
        )

        prompt = f"""
You are an AI assistant for Panchayat government records.

Answer the question ONLY using the supplied document.

IMPORTANT RULES:
1. Give ONLY the answer.
2. Do NOT explain.
3. Do NOT repeat the document.
4. Do NOT show the retrieved context.
5. Do NOT say "according to the document".
6. If the question asks for a number, return only the number.
7. If the question asks for a date, return only the date.
8. If the answer is not present, say:
Information not available in the document.

Question:
{question}

Document:
{context_text}
"""

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer Panchayat document questions "
                        "with only the exact answer."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = response.choices[0].message.content.strip()

        return answer

    # --------------------------------------------------------
    # LOCAL FALLBACK
    # --------------------------------------------------------

    # Groq is not configured.
    # Extract exact answer instead of returning full paragraph.

    best_context = contexts[0]["chunk"]

    answer = extract_exact_answer(
        question,
        best_context
    )

    if answer:
        return answer

    return "Information not available in the document."