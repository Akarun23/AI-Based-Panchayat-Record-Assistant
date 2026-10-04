import os
import cv2
import numpy as np
import pytesseract

# Change this only if Tesseract is not available in your PATH.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def preprocess_image(image_path: str):
    """Preprocess a scanned document for improved OCR."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the uploaded image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise while preserving document text.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive thresholding works well for unevenly lit scans.
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return processed


def extract_text(image_path: str, languages: str = "eng+tam") -> str:
    """Extract text using Tesseract OCR."""
    processed = preprocess_image(image_path)

    try:
        text = pytesseract.image_to_string(
            processed,
            lang=languages,
            config="--oem 3 --psm 6",
        )
    except pytesseract.TesseractError as exc:
        # If Tamil language data is unavailable, retry with English.
        if "Failed loading language" in str(exc) or "tam" in str(exc):
            text = pytesseract.image_to_string(
                processed,
                lang="eng",
                config="--oem 3 --psm 6",
            )
        else:
            raise

    return text.strip()
