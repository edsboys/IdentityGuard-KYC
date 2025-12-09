import pytesseract
import cv2
import numpy as np
import os

# 1. TELL PYTHON WHERE TESSERACT IS INSTALLED (CRITICAL FOR WINDOWS)
# If you installed it in a different drive, update this path.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path: str):
    """
    Reads an image and returns the text found on it.
    """
    try:
        # 2. Load the image using OpenCV
        img = cv2.imread(image_path)

        if img is None:
            return {"status": "error", "message": "Could not read image file"}

        # 3. Pre-processing (Make it easier for AI to read)
        # Convert to grayscale (Black & White)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thresholding (Make text pop out - purely black/white, no grey)
        # This helps if the ID photo has bad lighting
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # 4. Extract Text
        text = pytesseract.image_to_string(thresh)

        # Clean up the text (remove empty lines)
        clean_text = [line.strip() for line in text.split('\n') if line.strip()]

        return {
            "status": "success",
            "raw_text": text,
            "extracted_data": clean_text
        }

    except Exception as e:
        # Check if it's the specific "Tesseract not found" error
        if "No such file or directory" in str(e):
             return {"status": "error", "message": "Tesseract EXE not found. Did you install the Windows app?"}
        return {"status": "error", "message": str(e)}