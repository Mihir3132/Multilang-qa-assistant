import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    chroma_db_dir: str = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
    tesseract_cmd: str | None = os.getenv("TESSERACT_CMD") or None
    # OCR settings for scanned PDFs
    ocr_psm_mode: int = int(os.getenv("OCR_PSM_MODE", "6"))  # 6 = uniform block of text
    ocr_image_dpi: int = int(os.getenv("OCR_IMAGE_DPI", "300"))  # Higher DPI for better OCR


settings = Settings()

